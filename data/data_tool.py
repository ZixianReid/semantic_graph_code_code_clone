from data.graph_builder.code_graph import Sample

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
        edge_index_1 = [edge1[0], edge1[1]]
        edge_attr_1 = edge1[2]
        edge_index_2 = [edge2[0], edge2[1]]
        edge_attr_2 = edge2[2]
        if split_lable == 0:
            train_data.append([[x1, x2, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2], label])
        elif split_lable == 1:
            test_data.append([[x1, x2, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2], label])
        elif split_lable == 2:
            val_data.append([[x1, x2, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2], label])

    return train_data, test_data, val_data


def split_data_2(graph_dict: dict, labels: list, dataset_name: str):
    train_data = []
    test_data = []
    val_data = []
    for ele in labels:
        file_name_1 = ele[0]
        file_name_2 = ele[1]
        clone_label = int(ele[2])
        split_lable = int(ele[3])
        dataset_lable = int(ele[4])
        clone_type = str(ele[5])
        similarity_score = float(ele[6])
        x1, edge_index_1, edge_attr_1 = graph_dict[file_name_1][0]
        x2, edge_index_2, edge_attr_2 = graph_dict[file_name_2][0]
        if split_lable == 0:
            if dataset_name=='BigCloneBench' and dataset_lable==0:
                train_data.append(Sample(x1, x2, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2, clone_label, dataset_lable, clone_type, similarity_score))
        elif split_lable == 1:
            test_data.append(Sample(x1, x2, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2, clone_label, dataset_lable, clone_type, similarity_score))
        elif split_lable == 2:
            if dataset_name=='BigCloneBench' and dataset_lable==0:
                val_data.append(Sample(x1, x2, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2, clone_label, dataset_lable, clone_type, similarity_score))

    return train_data, test_data, val_data