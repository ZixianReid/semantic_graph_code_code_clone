from util.setting import log
from nets.load_net import gnn_model
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from eval.load_eval import evaluation
import time
def transfer_label(label):
    if label == 0:
        return -1
    else:
        return 1


def load_and_evaluate_model(model_path, dataset, params, net_params):
            device = net_params['device']
            model = gnn_model('graph_match_nerual_network', net_params)  # Replace 'MODEL_NAME' with the actual model name
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()  # Set the model to evaluation mode

            log.info(f"Loaded model from {model_path}")
            start_time = time.time()    
            for data in tqdm(dataset, desc="Processing Dataset", total=len(dataset)):
                x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2, label = data.x1, data.x2, data.edge_index_1, data.edge_index_2, data.edge_attr_1, data.edge_attr_2, data.clone_label
                label = transfer_label(label)
                label=torch.tensor(label, dtype=torch.float, device=device)
                x1=torch.tensor(x1, dtype=torch.long, device=device)
                x2=torch.tensor(x2, dtype=torch.long, device=device)
                edge_index1=torch.tensor(edge_index1, dtype=torch.long, device=device)
                edge_index2=torch.tensor(edge_index2, dtype=torch.long, device=device)
                edge_attr1=torch.tensor(edge_attr1, dtype=torch.long, device=device)
                edge_attr2=torch.tensor(edge_attr2, dtype=torch.long, device=device)   
                data_input=[x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
                prediction=model(data_input)

            end_time = time.time()
            log.info(f"Total time taken: {end_time - start_time}")
            print(f"Total time taken: {end_time - start_time}")
            log.info("Evaluation completed")

def evaluation_gmn(model, dataset, params, net_params):
    device = net_params['device']
    bcb_samples = []
    gcj_samples = []
    gpt_samples = []
    results=[]
    for data in tqdm(dataset, desc="Processing Dataset", total=len(dataset)):
        x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2, label = data.x1, data.x2, data.edge_index_1, data.edge_index_2, data.edge_attr_1, data.edge_attr_2, data.clone_label
        label = transfer_label(label)
        label=torch.tensor(label, dtype=torch.float, device=device)
        x1=torch.tensor(x1, dtype=torch.long, device=device)
        x2=torch.tensor(x2, dtype=torch.long, device=device)
        edge_index1=torch.tensor(edge_index1, dtype=torch.long, device=device)
        edge_index2=torch.tensor(edge_index2, dtype=torch.long, device=device)
        edge_attr1=torch.tensor(edge_attr1, dtype=torch.long, device=device)
        edge_attr2=torch.tensor(edge_attr2, dtype=torch.long, device=device)
        data_input=[x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
        prediction=model(data_input)
        output=F.cosine_similarity(prediction[0],prediction[1])
        results.append(output.item())
        prediction = torch.sign(output).item()

        if prediction>params['threshold']:
            prediction=int(1)
        else:
            prediction=int(0)

        dataset_label = data.dataset_label
        if int(dataset_label) == 0:
            bcb_samples.append((data, prediction))
        elif int(dataset_label) == 1:
             gcj_samples.append((data, prediction))
        elif int(dataset_label) == 2:
            gpt_samples.append((data, prediction))

    
    evaluation(bcb_samples, gcj_samples, gpt_samples)
    #     if prediction>params['threshold'] and label.item()==1:
    #         tp+=1
    #         #print('tp')
    #     if prediction<=params['threshold'] and label.item()==-1:
    #         tn+=1
    #         #print('tn')
    #     if prediction>params['threshold'] and label.item()==-1:
    #         fp+=1
    #         #print('fp')
    #     if prediction<=params['threshold'] and label.item()==1:
    #         fn+=1
    # precision=tp/(tp+fp)
    # recall=tp/(tp+fn)
    # f1=2*precision*recall/(precision+recall)
    # log.info(f"Precision: {precision}")
    # log.info(f"Recall: {recall}")
    # log.info(f"F1: {f1}")




   
def train_gmn(MODEL_NAME, dataset, params, net_params, dirs):
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs

    def create_batches(data):
        batches = [data[graph:graph + params['batch_size']] for graph in range(0, len(data), params['batch_size'])]
        return batches

    vocablen, trainset, valset, testset = dataset.vocab_length, dataset.train_data, dataset.val_data, dataset.test_data
    device = net_params['device']
    net_params['vocablen'] = vocablen
    log.info(f"Vocab length: {vocablen}")
    log.info(f"Trainset length: {len(trainset)}")
    log.info(f"Valset length: {len(valset)}")
    log.info(f"Testset length: {len(testset)}")

    # Model setting
    log.info("Model setting")
    model = gnn_model(MODEL_NAME, net_params)
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['lr_reduce_factor'], patience=params['lr_schedule_patience'], verbose=True)

    # Loss functions
    criterion = nn.CosineEmbeddingLoss()
    criterion2 = nn.MSELoss()

    epochs = trange(params['epochs'], leave=True, desc="Epoch")
    for epoch in epochs:
        model.train()  # Set the model to training mode
        batches = create_batches(trainset)
        total_loss = 0.0
        main_index = 0.0

        for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
            optimizer.zero_grad()
            batchloss = 0
            for data in batch:
                x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2, label = data.x1, data.x2, data.edge_index_1, data.edge_index_2, data.edge_attr_1, data.edge_attr_2, data.clone_label
                label = transfer_label(label)
                label = torch.tensor(label, dtype=torch.float, device=device)
                x1 = torch.tensor(x1, dtype=torch.long, device=device)
                x2 = torch.tensor(x2, dtype=torch.long, device=device)
                edge_index1 = torch.tensor(edge_index1, dtype=torch.long, device=device)
                edge_index2 = torch.tensor(edge_index2, dtype=torch.long, device=device)
                edge_attr1 = torch.tensor(edge_attr1, dtype=torch.long, device=device)
                edge_attr2 = torch.tensor(edge_attr2, dtype=torch.long, device=device)
                data = [x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
                prediction = model(data)
                cossim = F.cosine_similarity(prediction[0], prediction[1])
                batchloss = batchloss + criterion2(cossim, label)
            
            batchloss.backward(retain_graph=True)
            optimizer.step()
            loss = batchloss.item()
            total_loss += loss
            main_index += len(batch)
            loss = total_loss / main_index
            epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))

        # Validation loss computation for scheduler
        log.info(f"Validating model at epoch {epoch}")
        val_loss = 0.0
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for val_data in valset:
                x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2, label = val_data.x1, val_data.x2, val_data.edge_index_1, val_data.edge_index_2, val_data.edge_attr_1, val_data.edge_attr_2, val_data.clone_label
                label = transfer_label(label)
                label = torch.tensor(label, dtype=torch.float, device=device)
                x1 = torch.tensor(x1, dtype=torch.long, device=device)
                x2 = torch.tensor(x2, dtype=torch.long, device=device)
                edge_index1 = torch.tensor(edge_index1, dtype=torch.long, device=device)
                edge_index2 = torch.tensor(edge_index2, dtype=torch.long, device=device)
                edge_attr1 = torch.tensor(edge_attr1, dtype=torch.long, device=device)
                edge_attr2 = torch.tensor(edge_attr2, dtype=torch.long, device=device)
                data = [x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
                prediction = model(data)
                cossim = F.cosine_similarity(prediction[0], prediction[1])
                val_loss += criterion2(cossim, label).item()

        val_loss /= len(valset)  # Average validation loss
        scheduler.step(val_loss)  # Update the learning rate based on validation loss

        log.info(f"Epoch {epoch}, Validation Loss: {val_loss}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        # Periodic evaluation and saving
        if epoch % params['eval_epoch_interval'] == 0:
            log.info(f"Start evaluation on testset in epoch: {epoch}")
            evaluation_gmn(model, testset, params, net_params)
        if epoch % params['save_epoch_interval'] == 0:
            log.info(f"Saving model in epoch: {epoch}")
            torch.save(model.state_dict(), f"{root_ckpt_dir}/model_{epoch}.pth")

    # # Final test evaluation
    # log.info(f"Start evaluation on testset in epoch: {epoch}")
    # evaluation_gmn(model, testset, params, net_params)



