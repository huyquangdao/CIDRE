"""[summary]"""
import torch
from nltk.corpus import wordnet as wn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm_notebook


ner_vocab = {"O": 0, "B_Chemical": 1, "I_Chemical": 2, "B_Disease": 3, "I_Disease": 4}
ner_idx2label = {0: "O", 1: "B_Chemical", 2: "I_Chemical", 3: "B_Disease", 4: "I_Disease"}


def pad_sequences(list_token_ids, max_length, pad_idx):
    padded_list_token_ids = []
    padded_seq_mask = []
    for token_ids in list_token_ids:
        pad_length = max_length - len(token_ids)
        token_masked = [1] * len(token_ids) + [0] * pad_length
        padded_token_ids = token_ids + [pad_idx] * pad_length
        padded_list_token_ids.append(padded_token_ids)
        padded_seq_mask.append(token_masked)
    return padded_list_token_ids, padded_seq_mask


def pad_nodes(batch_list_nodes_idx, max_length, max_node, pad_idx=15):

    padded_list_nodes = []
    padded_nodes_masked = []
    # loop over batch of sentences
    for list_nodes_ids in batch_list_nodes_idx:

        sent_node_ids = []
        sent_nodes_masked = []

        for node_ids in list_nodes_ids:
            pad_lenght = max_node - len(node_ids)
            node_masked = [1] * len(node_ids) + [0] * pad_lenght
            padded_node_ids = node_ids + [pad_idx] * pad_lenght
            sent_node_ids.append(padded_node_ids)
            sent_nodes_masked.append(node_masked)

        pad_sent_length = max_length - len(list_nodes_ids)

        padded_node_ids = [[pad_idx] * max_node] * pad_sent_length
        padded_node_masked = [[0] * max_node] * pad_sent_length

        padded_sent_node_ids = sent_node_ids + padded_node_ids
        padded_sent_nodes_masked = sent_nodes_masked + padded_node_masked

        padded_list_nodes.append(padded_sent_node_ids)
        padded_nodes_masked.append(padded_sent_nodes_masked)

    return padded_list_nodes, padded_nodes_masked


def pad_entity(batch_list_entity_map, max_mentions, max_entity_span, pad_idx=15):

    padded_entities = []
    padded_entities_span_masked = []

    for sent_entity_map in batch_list_entity_map:

        padded_sent_entity_map = []
        padded_sent_entity_span_masked = []

        for mention in sent_entity_map:
            pad_length = max_entity_span - len(mention)

            if pad_length >= 0:
                padded_mention = mention + [pad_idx] * pad_length
                padded_masked = [1] * len(mention) + [0] * pad_length
            else:
                padded_masked = [1] * len(mention[:max_entity_span])
                padded_mention = mention[:max_entity_span]

            padded_sent_entity_map.append(padded_mention)
            padded_sent_entity_span_masked.append(padded_masked)

        pad_entity_mention = max_mentions - len(sent_entity_map)
        padded_sent_entity_span_masked = padded_sent_entity_span_masked + [[0] * max_entity_span] * pad_entity_mention
        padded_sent_entity_map = padded_sent_entity_map + [[pad_idx] * max_entity_span] * pad_entity_mention

        padded_entities.append(padded_sent_entity_map)
        padded_entities_span_masked.append(padded_sent_entity_span_masked)

    return padded_entities, padded_entities_span_masked


def pad_characters(list_char_ids, batch_max_length, max_char_length, char_pad_idx):

    # print('max char length: ',max_char_length)

    padded_char_ids = []

    for doc_char_ids in list_char_ids:
        pad_length = batch_max_length - len(doc_char_ids)
        doc_padded_char_ids = []
        for token_chars in doc_char_ids:

            char_pad_length = max_char_length - len(token_chars)
            pad = [char_pad_idx] * char_pad_length
            padded_token_chars = token_chars + pad

            assert len(padded_token_chars) == max_char_length
            doc_padded_char_ids.append(padded_token_chars)

        padded_batch_chars = [[char_pad_idx] * max_char_length] * pad_length
        doc_padded_char_ids = doc_padded_char_ids + padded_batch_chars
        padded_char_ids.append(doc_padded_char_ids)

    return padded_char_ids


def pad_tensor(list_tensor, batch_length):

    list_padded_tensor = []
    for tensor in list_tensor:
        pad_length = batch_length - tensor.shape[1]
        pad_tensor = torch.zeros(size=(pad_length, tensor.shape[-1])).cuda()

        padded_tensor = torch.cat([tensor.squeeze(0), pad_tensor], dim=0)

        assert padded_tensor.shape[0] == batch_length

        list_padded_tensor.append(padded_tensor)
    return list_padded_tensor


class CDRDataset(Dataset):
    def __init__(
        self,
        all_doc_token_ids,
        all_in_nodes_idx,
        all_out_nodes_idx,
        all_in_edge_label_ids,
        all_out_edge_label_ids,
        all_pos_ids,
        all_char_ids,
        all_elmo,
        all_entity_mapping,
        all_ner_label_ids,
        labels,
        model_type,
        max_node_in=5,
        max_node_out=30,
        max_mentions=30,
        max_entity_span=20,
        max_char_length=96,
    ):

        super(CDRDataset, self).__init__()

        self.all_doc_token_ids = all_doc_token_ids

        self.all_in_nodes_idx = all_in_nodes_idx
        self.all_in_edge_label_ids = all_in_edge_label_ids

        self.all_out_nodes_idx = all_out_nodes_idx
        self.all_out_edge_label_ids = all_out_edge_label_ids
        self.all_pos_ids = all_pos_ids
        self.all_char_ids = all_char_ids

        self.all_ner_label_ids = all_ner_label_ids
        self.labels = labels
        self.all_entity_mapping = all_entity_mapping
        # self.all_flair = all_flair

        self.max_node_in = max_node_in
        self.max_node_out = max_node_out
        self.max_mentions = max_mentions
        self.max_entity_span = max_entity_span
        self.max_char_length = max_char_length

        self.all_elmo = all_elmo
        self.ner_vocab = ner_vocab

        self.node_pad_idx = 4
        self.entity_pad_idx = 5
        self.edge_pad_idx = 10

        self.model_type = model_type
        # self.max_distant = 200

    def __len__(self):
        return len(self.labels)

    def collate_fn(self, batch):
        (
            list_token_ids,
            list_in_nodes_idx,
            list_out_nodes_idx,
            list_in_edge_label_ids,
            list_out_edge_label_ids,
            list_pos_ids,
            list_char_ids,
            list_elmo,
            list_chem_en_map,
            list_dis_en_map,
            list_ner_label_ids,
            list_label_ids,
        ) = list(zip(*batch))

        batch_max_length = -1
        for token_ids in list_token_ids:
            batch_max_length = max(len(token_ids), batch_max_length)

        # ok
        padded_list_token_ids, padded_list_token_mask = pad_sequences(
            list_token_ids, batch_max_length, self.word_vocab["<PAD>"]
        )
        padded_list_pos_ids, _ = pad_sequences(list_pos_ids, batch_max_length, self.pos_vocab["<PAD>"])

        padded_list_ner_ids, _ = pad_sequences(list_ner_label_ids, batch_max_length, self.ner_vocab["O"])

        # ok
        padded_list_char_ids = pad_characters(
            list_char_ids, batch_max_length, self.max_char_length, self.char_vocab["<PAD>"]
        )

        padded_list_elmo_tensors = pad_tensor(list_elmo, batch_max_length)
        # padded_list_flair_tensors = pad_tensor(list_flair, batch_max_length)

        # ok
        padded_in_nodes_idx, padded_in_nodes_mask = pad_nodes(
            list_in_nodes_idx, batch_max_length, self.max_node_in, self.node_pad_idx
        )
        padded_out_nodes_idx, padded_out_nodes_mask = pad_nodes(
            list_out_nodes_idx, batch_max_length, self.max_node_out, self.node_pad_idx
        )

        # ok
        padded_chem_entity_map, padded_chem_entity_mask = pad_entity(
            list_chem_en_map, self.max_mentions, self.max_entity_span, self.entity_pad_idx
        )

        # ok
        padded_dis_entity_map, padded_dis_entity_mask = pad_entity(
            list_dis_en_map, self.max_mentions, self.max_entity_span, self.entity_pad_idx
        )

        padded_in_edges_idx, padded_in_edges_mask = pad_nodes(
            list_in_edge_label_ids, batch_max_length, self.max_node_in, self.rel_vocab["<PAD>"]
        )
        padded_out_edges_idx, padded_out_edges_mask = pad_nodes(
            list_out_edge_label_ids, batch_max_length, self.max_node_out, self.rel_vocab["<PAD>"]
        )

        token_ids_tensor = torch.LongTensor(padded_list_token_ids)
        token_ids_mask_tensor = torch.LongTensor(padded_list_token_mask)

        pos_ids_tensor = torch.LongTensor(padded_list_pos_ids)
        char_ids_tensor = torch.LongTensor(padded_list_char_ids)

        ner_label_ids_tensor = torch.LongTensor(padded_list_ner_ids)
        elmo_tensor = torch.stack(padded_list_elmo_tensors)

        elmo_tensor.require_grad = False

        in_nodes_idx_tensor = torch.LongTensor(padded_in_nodes_idx)
        in_nodes_mask_tensor = torch.LongTensor(padded_in_nodes_mask)

        out_nodes_idx_tensor = torch.LongTensor(padded_out_nodes_idx)
        out_nodes_mask_tensor = torch.LongTensor(padded_out_nodes_mask)

        # batch_size, max_mentions, max_span
        chem_entity_map_tensor = torch.LongTensor(padded_chem_entity_map)
        chem_entity_map_mask_tensor = torch.LongTensor(padded_chem_entity_mask)

        dis_entity_map_tensor = torch.LongTensor(padded_dis_entity_map)
        dis_entity_map_mask_tensor = torch.LongTensor(padded_dis_entity_mask)

        chem_start_idx = chem_entity_map_tensor[..., 0]
        dis_start_idx = dis_entity_map_tensor[..., 0]

        start_distant = torch.abs(
            chem_start_idx.unsqueeze(2).repeat(1, 1, self.max_mentions)
            - dis_start_idx.unsqueeze(1).repeat(1, self.max_mentions, 1)
        )

        in_edge_idx_tensor = torch.LongTensor(padded_in_edges_idx)
        in_edge_idx_mask = torch.LongTensor(padded_in_edges_mask)

        out_edge_idx_tensor = torch.LongTensor(padded_out_edges_idx)
        out_edge_idx_mask = torch.LongTensor(padded_out_edges_mask)

        label_ids_tensor = torch.LongTensor(list_label_ids)

        return (
            token_ids_tensor,
            token_ids_mask_tensor,
            pos_ids_tensor,
            char_ids_tensor,
            elmo_tensor,
            in_nodes_idx_tensor,
            in_nodes_mask_tensor,
            out_nodes_idx_tensor,
            out_nodes_mask_tensor,
            in_edge_idx_tensor,
            in_edge_idx_mask,
            out_edge_idx_tensor,
            out_edge_idx_mask,
            chem_entity_map_tensor,
            chem_entity_map_mask_tensor,
            dis_entity_map_tensor,
            dis_entity_map_mask_tensor,
            start_distant,
            ner_label_ids_tensor,
            label_ids_tensor,
        )

    def set_vocabs(self, word_vocab, rel_vocab, pos_vocab, char_vocab):
        self.word_vocab = word_vocab
        self.rel_vocab = rel_vocab
        self.pos_vocab = pos_vocab
        self.char_vocab = char_vocab

        self.idx2word = {k: v for v, k in word_vocab.items()}

    def __getitem__(self, idx):
        

        label = self.labels[idx]
        pud_id, c_id, d_id, rel = label

        if rel == "CID":
            label_ids = 1
        else:
            label_ids = 0

        if self.model_type == "full":

            token_ids = self.all_doc_token_ids[pud_id]
            # token_texts = [self.idx2word[x] for x in token_ids]

            in_nodes_idx = self.all_in_nodes_idx[pud_id]
            in_edge_label_ids = self.all_in_edge_label_ids[pud_id]
            out_nodes_idx = self.all_out_nodes_idx[pud_id]
            out_edge_label_ids = self.all_out_edge_label_ids[pud_id]
            pos_ids = self.all_pos_ids[pud_id]
            char_ids = self.all_char_ids[pud_id]

            assert len(char_ids) == len(pos_ids) == len(token_ids)

            chem_entity_map = self.all_entity_mapping[pud_id][c_id]
            dis_entity_map = self.all_entity_mapping[pud_id][d_id]

            elmo_embedding = self.all_elmo[pud_id]
            ner_label_ids = self.all_ner_label_ids[pud_id]
        
        elif self.model_type == "inter":

            token_ids = self.all_doc_token_ids[label]
            # token_texts = [self.idx2word[x] for x in token_ids]

            in_nodes_idx = self.all_in_nodes_idx[label]
            in_edge_label_ids = self.all_in_edge_label_ids[label]
            out_nodes_idx = self.all_out_nodes_idx[label]
            out_edge_label_ids = self.all_out_edge_label_ids[label]
            pos_ids = self.all_pos_ids[label]
            char_ids = self.all_char_ids[label]

            assert len(char_ids) == len(pos_ids) == len(token_ids)

            chem_entity_map = self.all_entity_mapping[label][c_id]
            dis_entity_map = self.all_entity_mapping[label][d_id]

            elmo_embedding = self.all_elmo[label].cuda()
            ner_label_ids = self.all_ner_label_ids[label]

        return (
            token_ids,
            in_nodes_idx,
            out_nodes_idx,
            in_edge_label_ids,
            out_edge_label_ids,
            pos_ids,
            char_ids,
            elmo_embedding,
            chem_entity_map,
            dis_entity_map,
            ner_label_ids,
            label_ids,
        )