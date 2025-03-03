import torch
from dataset import OSICData_train, OSICData_test
from model import TabCT
import numpy as np
import pandas as pd 
from tqdm import tqdm  # Displays progress bars for loops and tasks
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from configs import HyperParameters

params = HyperParameters("slope_train_vit_simple")

train = pd.read_csv(f'{params.data_folder}/train.csv')

# device
gpu = torch.device(f"cuda:{params.gpu_index}" if torch.cuda.is_available() else "cpu")

# Function to generate tabular features from a DataFrame for a single patient
def get_tab(df):
    # Initialize a feature vector with the normalized age of the patient
    # Normalization is done using the mean and standard deviation of the training data's Age
    vector = [(df.Age.values[0] - train.Age.values.mean()) / train.Age.values.std()]
    
    # Add a binary encoding for the patient's gender
    # 0 for 'Male', 1 for 'Female'
    if df.Sex.values[0] == 'Male':
        vector.append(0)
    else:
        vector.append(1)
    
    # Add a one-hot encoded representation for the patient's smoking status
    # 'Never smoked' -> [0, 0]
    # 'Ex-smoker' -> [1, 1]
    # 'Currently smokes' -> [0, 1]
    # Any other status -> [1, 0]
    if df.SmokingStatus.values[0] == 'Never smoked':
        vector.extend([0, 0])
    elif df.SmokingStatus.values[0] == 'Ex-smoker':
        vector.extend([1, 1])
    elif df.SmokingStatus.values[0] == 'Currently smokes':
        vector.extend([0, 1])
    else:
        vector.extend([1, 0])
    
    # Convert the feature vector to a NumPy array and return it
    return np.array(vector)

# Initialize dictionaries and list to store results
A = {}  # Dictionary to store the slopes (linear regression results) for each patient
TAB = {}  # Dictionary to store tabular features for each patient
P = []  # List to store patient IDs

# Iterate through each unique patient in the training dataset
for i, p in enumerate(tqdm(train.Patient.unique())):
    # Filter data for the current patient
    sub = train.loc[train.Patient == p, :] 
    
    # Extract Forced Vital Capacity (FVC) and Weeks (time points) for the patient
    fvc = sub.FVC.values
    weeks = sub.Weeks.values
    
    # For each patient, the code calculates the slope of FVC over time (Weeks) using linear regression, generates tabular features, and stores the results (slope, features, and patient ID) in dictionaries and a list.
    c = np.vstack([weeks, np.ones(len(weeks))]).T  # Add a column of ones for the intercept term

    a, _ = np.linalg.lstsq(c, fvc, rcond=None)[0]  # Returns the slope 'a' of the best fit line
    # ref: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html

    A[p] = a
    TAB[p] = get_tab(sub)
    P.append(p)



def score(fvc_true, fvc_pred, sigma):
    sigma_clip = np.maximum(sigma, 70)
    delta = np.abs(fvc_true - fvc_pred)
    delta = np.minimum(delta, 1000)
    sq2 = np.sqrt(2)
    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)
    return np.mean(metric)


def score_avg(p, a): # patient id, predicted a
    percent_true = train.Percent.values[train.Patient == p]
    fvc_true = train.FVC.values[train.Patient == p]
    weeks_true = train.Weeks.values[train.Patient == p]

    fvc = a * (weeks_true - weeks_true[0]) + fvc_true[0]
    percent = percent_true[0] - a * abs(weeks_true - weeks_true[0])
    return score(fvc_true, fvc, percent)

def rmse_avg(p, a): # patient id, predicted a
    percent_true = train.Percent.values[train.Patient == p]
    fvc_true = train.FVC.values[train.Patient == p]
    weeks_true = train.Weeks.values[train.Patient == p]

    fvc = a * (weeks_true - weeks_true[0]) + fvc_true[0]
    return root_mean_squared_error(fvc_true, fvc)



def smape(targets, outs):
    denominator = (np.abs(targets) + np.abs(outs)) / 2
    ape = np.abs(targets - outs) / denominator
    return np.mean(ape) * 100

train_s, test = train_test_split(P, test_size=0.2, random_state=57)

# removing noisy data
P = [p for p in P if p not in ['ID00011637202177653955184', 'ID00052637202186188008618']] # in bayad bere bala

# ==================== Train ====================
print("==================== Train ====================")
for model in params.train_models:
    log = open(f"{params.result_dir}/september8th/{model}simplestep2hybrid.txt", "a+")
    kfold =KFold(n_splits=params.nfold)
    
    ifold = 0
    min_sq = {}

    for train_index, valid_index in kfold.split(train_s):  
        
        p_train = np.array(P)[train_index] 
        p_valid = np.array(P)[valid_index] 
        print(len(p_train))
        osic_train = OSICData_train(p_train, A, TAB)
        train_loader = torch.utils.data.DataLoader(osic_train, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, drop_last=True)

        osic_val = OSICData_test(p_valid, A, TAB)
        val_loader = torch.utils.data.DataLoader(osic_val, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)

        tabct = TabCT(cnn = model).to(gpu)
        print(f"creating {model}")
        print(f"fold: {ifold}")
        log.write(f"fold: {ifold}\n")

        

        n_epochs = params.n_epochs # max 30 epochs, patience 5, find the suitable epoch number for later final training

        best_epoch = n_epochs # 30


        optimizer = torch.optim.AdamW(tabct.parameters())
        criterion = torch.nn.L1Loss()

        max_score = 99999999.0000 # here, max score ]= minimum score
        tot_rmse = []
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            tabct.train()

            tabular_loss = []
            for i, data in enumerate(tqdm(train_loader, 0)):

                [mask, x, t], a, _ = data

                x = x.to(gpu)
                mask = mask.to(gpu)
                t = t.to(gpu)
                a = a.to(gpu)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward + backward + optimize
                outputs, tab_loss = tabct(x, t, mask) # here
                # print(outputs.size())
                tabular_loss.append(tab_loss)
                loss = criterion(outputs, a)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            print(f"tabular loss: {tabular_loss}")
            print(f"epoch {epoch+1} train: {running_loss}")
            log.write(f"epoch {epoch+1} train: {running_loss}\n")


            running_loss = 0.0
            pred_a = {}
            tabct.eval()
            tabular_loss = []
            for i, data in enumerate(tqdm(val_loader, 0)):

                [mask, x, t], a, pid = data

                x = x.to(gpu)
                mask = mask.to(gpu)
                t = t.to(gpu)
                a = a.to(gpu)

                # forward
                outputs, tab_loss = tabct(x, t, mask)
                loss = criterion(outputs, a)
                tabular_loss.append(tab_loss)
                pids = pid
                preds_a = outputs.detach().cpu().numpy().flatten()

                for j, p_d in enumerate(pids):
                    pred_a[p_d] = preds_a[j]

               


                # print statistics
                running_loss += loss.item()
            print(tabular_loss)
            print(f"epoch {epoch+1} val: {running_loss}")
            log.write(f"epoch {epoch+1} val: {running_loss}\n")

            # totals
            tot_r2_score = []
            tot_mape_score = []
            tot_mae_score = []
            tot_smape_score = []

            # everyone
            score_v = 0.
            rmse = 0.
            test_r2_score = []
            test_mape_score = []
            test_mae_score = []
            test_smape_score = []
            print(len(p_valid))

            for p in p_valid:
                score_v += (score_avg(p, pred_a[p]))/len(p_valid)
                rmse += (rmse_avg(p, pred_a[p]))/len(p_valid)
                fvc_true = train.FVC.values[train.Patient == p]
                weeks_true = train.Weeks.values[train.Patient == p]
                fvc_pred = pred_a[p] * (weeks_true - weeks_true[0]) + fvc_true[0]

                test_r2_score.append(r2_score(fvc_true, fvc_pred))
                test_mape_score.append(np.mean(np.abs((fvc_true - fvc_pred) / fvc_true)) * 100)
                test_mae_score.append(np.mean(np.abs(fvc_pred - fvc_true)))
                test_smape_score.append(smape(fvc_true, fvc_pred))
            #------------------------
            tot_rmse.append(rmse)
            tot_r2_score.append(np.asanyarray(test_r2_score))
            tot_mape_score.append(np.asanyarray(test_mape_score))
            tot_mae_score.append(np.asanyarray(test_mae_score))
            tot_smape_score.append(np.asanyarray(test_smape_score))
            #------------------------

            print("this is rmse")
            print(tot_rmse)
            print("this is r2")
            print(tot_r2_score)
            print("this is mape")
            print(tot_mape_score)
            print("this is mae")
            print(tot_mae_score)
            print("this is smape")
            print(tot_smape_score)
            print(f"val score: {score_v}")
            log.write(f"val score: {score_v}\n")
            log.write(f"val rmse: {rmse}\n")

            if score_v <= max_score:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': tabct.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'score': score_v
                    }, f"{params.results_dir}/september8th/{model}simplestep2hybrid.tar")
                max_score = score_v
                best_epoch = epoch + 1
        min_sq[ifold] = np.array(tot_rmse)

        ifold += 1

        # destroy model
        del tabct
        torch.cuda.empty_cache()

# ==================== Test ====================
print("==================== Test ====================")
test = [p for p in test if p not in ['ID00011637202177653955184', 'ID00052637202186188008618']]
osic_test = OSICData_test(test, A, TAB)
test_loader = torch.utils.data.DataLoader(osic_test, batch_size=1, num_workers=params.num_workers)


# load the best model
tabct = TabCT(cnn = model).to(gpu)
tabct.load_state_dict(torch.load(f"{params.results_dir}/september8th/{model}simplestep2hybrid.tar")["model_state_dict"])

running_loss = 0.0
pred_a = {}
tabct.eval()
tabular_loss = []
for i, data in enumerate(tqdm(test_loader, 0)):

    [mask, x, t], a, pid = data

    x = x.to(gpu)
    mask = mask.to(gpu)
    t = t.to(gpu)
    a = a.to(gpu)

    # forward
    outputs, tab_loss = tabct(x, t, mask)
    loss = criterion(outputs, a)
    tabular_loss.append(tab_loss)
    pids = pid
    preds_a = outputs.detach().cpu().numpy().flatten()
    print([outputs, pid])
    for j, p_d in enumerate(pids):
        pred_a[p_d] = preds_a[j]




    # print statistics
    running_loss += loss.item()
print(tabular_loss)


# totals
tot_r2_score = []
tot_mape_score = []
tot_mae_score = []
tot_smape_score = []

# everyone
score_v = 0.
rmse = 0.
test_r2_score = []
test_mape_score = []
test_mae_score = []
test_smape_score = []

for p in test:
    score_v += (score_avg(p, pred_a[p]))/len(test)
    rmse += (rmse_avg(p, pred_a[p]))/len(test)
    fvc_true = train.FVC.values[train.Patient == p]
    weeks_true = train.Weeks.values[train.Patient == p]
    fvc_pred = pred_a[p] * (weeks_true - weeks_true[0]) + fvc_true[0]

    test_r2_score.append(r2_score(fvc_true, fvc_pred))
    test_mape_score.append(np.mean(np.abs((fvc_true - fvc_pred) / fvc_true)) * 100)
    test_mae_score.append(np.mean(np.abs(fvc_pred - fvc_true)))
    test_smape_score.append(smape(fvc_true, fvc_pred))
#------------------------
tot_rmse.append(rmse)
tot_r2_score.append(np.asanyarray(test_r2_score))
tot_mape_score.append(np.asanyarray(test_mape_score))
tot_mae_score.append(np.asanyarray(test_mae_score))
tot_smape_score.append(np.asanyarray(test_smape_score))
#------------------------

print("this is rmse")
print(tot_rmse)
print("this is r2")
print(tot_r2_score)
print("this is mape")
print(tot_mape_score)
print("this is mae")
print(tot_mae_score)
print("this is smape")
print(tot_smape_score)
print(f"val score: {score_v}")
log.write(f"val score: {score_v}\n")
log.write(f"val rmse: {rmse}\n")

