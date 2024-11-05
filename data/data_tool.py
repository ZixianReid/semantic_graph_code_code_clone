

def split_data(graph_dict: dict, labels: list):
    # split data into train, test, val

    train_data = []
    test_data = []
    val_data = []
    for ele in labels:
        file_name_1 = ele[0]
        file_name_2 = ele[1]
        label = int(ele[2])
        split_lable = int(ele[3])

        x1, edge1 = graph_dict[file_name_1]
        x2, edge2 = graph_dict[file_name_2]
        if split_lable == 0:
            train_data.append([[x1, x2, edge1, edge2], label])
        elif split_lable == 1:
            test_data.append([[x1, x2, edge1, edge2], label])
        elif split_lable == 2:
            val_data.append([[x1, x2, edge1, edge2], label])

    return train_data, test_data, val_data

