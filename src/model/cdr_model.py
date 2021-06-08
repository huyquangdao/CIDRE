import torch
import torch.nn as nn


class GraphAttnModule(nn.Module):
    def __init__(self, n_heads, input_hidden_size):

        super(GraphAttnModule, self).__init__()
        self.weight_matrix = nn.Linear(input_hidden_size, input_hidden_size, bias=False)
        self.attn_matrix = nn.Parameter(torch.randn(n_heads, (2 * input_hidden_size) // n_heads, 1))
        self.leaky_relu = nn.LeakyReLU()

        self.n_heads = n_heads

    def forward(self, node_hidden_states, neibor_nodes_hidden_state, neibor_nodes_mask=None):
        # node_hidden_states = [b, max_seq_length, hidden_size]
        # neibor_nodes_hidden_state = [b, max_seq_length, max_node, hidden_size]
        # neibor_nodes_mask = [b, max_seq_length, max_node]

        bs, max_seq_length, max_node, _ = neibor_nodes_hidden_state.shape
        # applying linear transformation
        node_hidden_states = self.weight_matrix(node_hidden_states)
        neibor_nodes_hidden_state = self.weight_matrix(neibor_nodes_hidden_state)

        temp = neibor_nodes_hidden_state.clone().view(bs * max_seq_length, max_node, -1)
        temp = temp.view(bs * max_seq_length, max_node, self.n_heads, -1).permute(0, 2, 1, 3)
        # temp = [bs * max_seq_length, n_head, max_node, (hidden_state )// n_heads]
        # concat features
        node_hidden_states = node_hidden_states.unsqueeze(2).repeat(1, 1, max_node, 1)
        attn_input = torch.cat([node_hidden_states, neibor_nodes_hidden_state], dim=-1)

        attn_input = attn_input.view(bs * max_seq_length, max_node, -1)
        attn_input = attn_input.view(bs * max_seq_length, max_node, self.n_heads, -1).permute(0, 2, 1, 3)
        # attn_input = [bs * max_seq_length, n_heads, max_node, (hidden_size * 2) // n_heads]
        # self.weight_matrix = [n_heads, (hidden_size * 2) // n_heads, 1]
        attn_score = self.leaky_relu(torch.matmul(attn_input, self.attn_matrix.unsqueeze(0)))
        # attn_score = [bs * max_seq_length, n_heads,  max_node, 1]
        if neibor_nodes_mask is not None:
            attn_score = attn_score.masked_fill(
                neibor_nodes_mask.view(-1, max_node).unsqueeze(1).unsqueeze(-1) == 0, -1e10
            )

        attn_prob = torch.softmax(attn_score, dim=-2)
        # attn_prob = [bs * max_seq_length, n_heads,  max_node, 1]
        attn_output = torch.matmul(attn_prob.permute(0, 1, 3, 2), temp)
        # attn_output = [bs * max_seq_length, n_heads, 1, (hidden_size // n_heads)]

        attn_output = attn_output.squeeze(2).view(bs * max_seq_length, -1)
        attn_output = attn_output.view(bs, max_seq_length, -1)

        return attn_output


class GraphEncoder(nn.Module):
    def __init__(
        self,
        time_step,
        edge_vocab_size,
        pos_vocab_size,
        char_vocab_size,
        word_vocab_size,
        contextual_word_embedding_dim,
        edge_embedding_dim,
        pos_embedding_dim,
        combined_embedding_dim,
        lstm_hidden_size,
        hidden_size,
        in_attn_heads,
        out_attn_heads,
        use_char,
        use_pos,
        use_word,
        use_state,
        kernel_size=5,
        word_embedding_dim=200,
        n_filters=30,
        char_embedding_dim=30,
        use_attn=True,
        device=torch.device("cuda:0"),
        drop_out=0.2,
    ):

        super(GraphEncoder, self).__init__()

        self.edge_embedding = nn.Embedding(edge_vocab_size, edge_embedding_dim)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embedding_dim)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)

        self.n_filters = n_filters
        self.conv = nn.Conv1d(
            in_channels=char_embedding_dim, out_channels=n_filters, kernel_size=kernel_size, stride=1
        )
        self.drop_out = nn.Dropout(drop_out)
        self.char_embedding_dim = char_embedding_dim

        self.linear_node_edge = nn.Linear(2 * lstm_hidden_size + edge_embedding_dim, combined_embedding_dim)

        self.time_step = time_step
        self.device = device
        self.hidden_size = hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.use_char = use_char
        self.use_pos = use_pos
        self.use_word = use_word
        self.use_state = use_state

        if not self.use_word and self.use_char and self.use_pos:
            input_size = contextual_word_embedding_dim + n_filters + pos_embedding_dim
        elif not self.use_word and self.use_char and not self.use_pos:
            input_size = contextual_word_embedding_dim + n_filters
        elif not self.use_word and not self.use_char and self.use_pos:
            input_size = contextual_word_embedding_dim + pos_embedding_dim
        elif not self.use_word and not self.use_char and not self.use_pos:
            input_size = contextual_word_embedding_dim
        elif self.use_word:
            input_size = word_embedding_dim + n_filters + pos_embedding_dim

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            num_layers=1,
        )

        self.w_h = nn.Linear(2 * lstm_hidden_size, hidden_size, bias=False)
        self.w_cell = nn.Linear(2 * lstm_hidden_size, hidden_size, bias=False)

        self.w_in_ingate = nn.Linear(combined_embedding_dim, hidden_size, bias=False)
        self.u_in_ingate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_ingate = nn.Parameter(torch.zeros(hidden_size))
        self.w_out_ingate = nn.Linear(combined_embedding_dim, hidden_size, bias=False)
        self.u_out_ingate = nn.Linear(hidden_size, hidden_size, bias=False)

        self.w_in_forgetgate = nn.Linear(combined_embedding_dim, hidden_size, bias=False)
        self.u_in_forgetgate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_forgetgate = nn.Parameter(torch.zeros(hidden_size))
        self.w_out_forgetgate = nn.Linear(combined_embedding_dim, hidden_size, bias=False)
        self.u_out_forgetgate = nn.Linear(hidden_size, hidden_size, bias=False)

        self.w_in_outgate = nn.Linear(combined_embedding_dim, hidden_size, bias=False)
        self.u_in_outgate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_outgate = nn.Parameter(torch.zeros(hidden_size))
        self.w_out_outgate = nn.Linear(combined_embedding_dim, hidden_size, bias=False)
        self.u_out_outgate = nn.Linear(hidden_size, hidden_size, bias=False)

        self.w_in_cell = nn.Linear(combined_embedding_dim, hidden_size, bias=False)
        self.u_in_cell = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_cell = nn.Parameter(torch.zeros(hidden_size))
        self.w_out_cell = nn.Linear(combined_embedding_dim, hidden_size, bias=False)
        self.u_out_cell = nn.Linear(hidden_size, hidden_size, bias=False)

        # weight for attn
        self.W_g_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_g_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_g_in = nn.Parameter(torch.zeros(hidden_size))
        self.W_f_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_f_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_f_in = nn.Parameter(torch.zeros(hidden_size))

        self.W_o_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_o_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_o_in = nn.Parameter(torch.zeros(hidden_size))

        self.W_g_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_g_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_g_out = nn.Parameter(torch.zeros(hidden_size))
        self.W_f_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_f_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_f_out = nn.Parameter(torch.zeros(hidden_size))

        self.W_o_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_o_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_o_out = nn.Parameter(torch.zeros(hidden_size))

        self.W_in = nn.Linear(combined_embedding_dim, hidden_size)
        self.W_out = nn.Linear(combined_embedding_dim, hidden_size)

        self.use_attn = use_attn
        self.in_attn_module = GraphAttnModule(n_heads=in_attn_heads, input_hidden_size=hidden_size)
        self.out_attn_module = GraphAttnModule(n_heads=out_attn_heads, input_hidden_size=hidden_size)

    def collect_neighbor_representations(self, representations, positions):

        # representation = [batch, max_seg_length, hidden_dim]
        # positions = [batch, max_seq_length, max_node_to_collect]

        batch, max_seq_length, max_node_to_collect = positions.shape
        feature_dim = representations.shape[-1]
        positions = positions.view(batch, max_seq_length * max_node_to_collect)
        # positions = [batch, max_seq_length * max_node_to_collect ]

        collected_tensor = torch.gather(
            representations, 1, positions[..., None].expand(*positions.shape, representations.shape[-1])
        )
        collected_tensor = collected_tensor.view(batch, max_seq_length, max_node_to_collect, feature_dim)
        return collected_tensor

    def forward(self, inputs):

        (
            elmo_tensor,
            token_ids_tensor,
            token_ids_mask_tensor,
            pos_ids_tensor,
            char_ids_tensor,
            in_nodes_idx_tensor,
            in_nodes_mask_tensor,
            out_nodes_idx_tensor,
            out_nodes_mask_tensor,
            in_edge_idx_tensor,
            in_edge_idx_mask,
            out_edge_idx_tensor,
            out_edge_idx_mask,
        ) = inputs

        bs, batch_length = token_ids_tensor.shape
        max_char_length = char_ids_tensor.shape[2]

        word_embedded = self.word_embedding(token_ids_tensor)

        char_embedded = self.char_embedding(char_ids_tensor)
        # char_embedded = [batch_size, max_seq_length, max_char_length, char_embedding_dim]
        char_embedded = char_embedded.view(bs * batch_length, max_char_length, self.char_embedding_dim)
        conv_char_embedded = torch.tanh(self.conv(char_embedded.permute(0, 2, 1)))
        conv_char_embedded = torch.max(conv_char_embedded.permute(0, 2, 1), dim=1)[0]
        conv_char_embedded = conv_char_embedded.view(bs, batch_length, self.n_filters)

        pos_embedded = self.pos_embedding(pos_ids_tensor)
        # input representation
        if not self.use_word and self.use_char and self.use_pos:
            concat_word_pos_embedded = self.drop_out(
                torch.cat([elmo_tensor, conv_char_embedded, pos_embedded], dim=-1)
            )
        elif not self.use_word and self.use_char and not self.use_pos:
            concat_word_pos_embedded = self.drop_out(torch.cat([elmo_tensor, conv_char_embedded], dim=-1))
        elif not self.use_word and not self.use_char and self.use_pos:
            concat_word_pos_embedded = self.drop_out(torch.cat([elmo_tensor, pos_embedded], dim=-1))
        elif not self.use_word and not self.use_char and not self.use_pos:
            concat_word_pos_embedded = self.drop_out(elmo_tensor)
        elif self.use_word:
            concat_word_pos_embedded = self.drop_out(
                torch.cat([word_embedded, conv_char_embedded, pos_embedded], dim=-1)
            )

        # gather context information for the input sequence with bilstm
        lstm_output, (_, _) = self.lstm(concat_word_pos_embedded.permute(1, 0, 2))
        lstm_output = lstm_output.permute(1, 0, 2)

        # batch_size, max_seq_length, max_node_in, edge_embedding_dim
        in_edge_embedded = self.edge_embedding(in_edge_idx_tensor)
        # batch_size, max_seq_length, max_node_in, edge_embedding_dim + word_embedding_dim
        collected_in_word_embedded = self.collect_neighbor_representations(lstm_output, in_nodes_idx_tensor)
        in_embedded = torch.cat([collected_in_word_embedded, in_edge_embedded], dim=-1)
        # multiply with mask
        in_embedded = in_embedded * in_nodes_mask_tensor.unsqueeze(-1)
        # sum over 2nd dimension
        in_embedded = torch.sum(in_embedded, dim=2)
        # batch_size, max_seq_length, max_node_in, edge_embedding_dim
        out_edge_embedded = self.edge_embedding(out_edge_idx_tensor)
        # batch_size, max_seq_length, max_node_in, edge_embedding_dim + word_embedding_dim
        collected_out_word_embedded = self.collect_neighbor_representations(lstm_output, out_nodes_idx_tensor)
        out_embedded = torch.cat([collected_out_word_embedded, out_edge_embedded], dim=-1)

        # multiply with mask
        out_embedded = out_embedded * out_nodes_mask_tensor.unsqueeze(-1)
        # sum over dimension 2
        out_embedded = torch.sum(out_embedded, dim=2)

        # project to lower dimension and apply non linear function
        in_embedded = torch.tanh(self.linear_node_edge(in_embedded))
        out_embedded = torch.tanh(self.linear_node_edge(out_embedded))

        bs, max_seq_length, _ = in_embedded.shape
        # node_hidden = torch.zeros(size=(bs, max_seq_length, self.hidden_size)).to(self.device)
        # node_cell = torch.zeros(size=(bs, max_seq_length, self.hidden_size)).to(self.device)

        node_hidden = torch.tanh(self.w_h(lstm_output))
        node_cell = torch.tanh(self.w_cell(lstm_output))

        if self.use_state:
            for t in range(self.time_step):
                passage_in_edge_prev_hidden = self.collect_neighbor_representations(node_hidden, in_nodes_idx_tensor)
                passage_out_edge_prev_hidden = self.collect_neighbor_representations(node_hidden, out_nodes_idx_tensor)

                if not self.use_attn:

                    passage_in_edge_prev_hidden = passage_in_edge_prev_hidden * in_nodes_mask_tensor.unsqueeze(-1)
                    passage_in_edge_prev_hidden = torch.sum(passage_in_edge_prev_hidden, dim=2)

                    passage_out_edge_prev_hidden = passage_out_edge_prev_hidden * out_nodes_mask_tensor.unsqueeze(-1)
                    passage_out_edge_prev_hidden = torch.sum(passage_out_edge_prev_hidden, dim=2)

                else:
                    passage_in_edge_prev_hidden = passage_in_edge_prev_hidden * in_nodes_mask_tensor.unsqueeze(-1)
                    passage_in_edge_prev_hidden = self.in_attn_module(
                        node_hidden, passage_in_edge_prev_hidden, in_nodes_mask_tensor
                    )
                    passage_out_edge_prev_hidden = passage_out_edge_prev_hidden * out_nodes_mask_tensor.unsqueeze(-1)
                    passage_out_edge_prev_hidden = self.out_attn_module(
                        node_hidden, passage_out_edge_prev_hidden, out_nodes_mask_tensor
                    )

                passage_edge_ingate = torch.sigmoid(
                    self.w_in_ingate(in_embedded)
                    + self.u_in_ingate(passage_in_edge_prev_hidden)
                    + self.w_out_ingate(out_embedded)
                    + self.u_out_ingate(passage_out_edge_prev_hidden)
                    + self.b_ingate
                )

                passage_edge_forgetgate = torch.sigmoid(
                    self.w_in_forgetgate(in_embedded)
                    + self.u_in_forgetgate(passage_in_edge_prev_hidden)
                    + self.w_out_forgetgate(out_embedded)
                    + self.u_out_forgetgate(passage_out_edge_prev_hidden)
                    + self.b_forgetgate
                )

                passage_edge_outgate = torch.sigmoid(
                    self.w_in_outgate(in_embedded)
                    + self.u_in_outgate(passage_in_edge_prev_hidden)
                    + self.w_out_outgate(out_embedded)
                    + self.u_out_outgate(passage_out_edge_prev_hidden)
                    + self.b_outgate
                )

                passage_edge_cell_input = torch.sigmoid(
                    self.w_in_cell(in_embedded)
                    + self.u_in_cell(passage_in_edge_prev_hidden)
                    + self.w_out_cell(out_embedded)
                    + self.u_out_cell(passage_out_edge_prev_hidden)
                    + self.b_cell
                )

                passage_edge_cell = passage_edge_forgetgate * node_cell + passage_edge_ingate * passage_edge_cell_input
                passage_edge_hidden = passage_edge_outgate * torch.tanh(passage_edge_cell)
                # node mask
                # [batch_size, passage_len, neighbor_vector_dim]
                node_cell = passage_edge_cell * token_ids_mask_tensor.unsqueeze(-1)
                node_hidden = passage_edge_hidden * token_ids_mask_tensor.unsqueeze(-1)

            return node_cell, node_hidden, lstm_output
        else:
            return (
                node_cell,
                lstm_output,
                lstm_output,
            )


class GraphStateLSTM(nn.Module):
    def __init__(
        self,
        relation_classes,
        ner_classes,
        encoder,
        entity_hidden_size,
        distance_embedding_dim,
        use_state,
        use_ner=False,
        distance_thresh=0,
        max_distance=600,
        drop_out=0.2,
    ):

        super(GraphStateLSTM, self).__init__()
        self.encoder = encoder
        self.relation_classes = relation_classes
        self.ner_classes = ner_classes
        self.use_ner = use_ner
        self.use_state = use_state

        self.encoder_hidden_size = encoder.hidden_size
        self.encoder_lstm_hidden_size = encoder.lstm_hidden_size
        self.distance_embedding_dim = distance_embedding_dim
        self.entity_hidden_size = entity_hidden_size
        self.distance_thresh = distance_thresh

        if self.use_state:
            self.linear_chem = nn.Linear(self.encoder_hidden_size, entity_hidden_size)
            self.linear_dis = nn.Linear(self.encoder_hidden_size, entity_hidden_size)
        else:
            self.linear_chem = nn.Linear(self.encoder_lstm_hidden_size * 2, entity_hidden_size)
            self.linear_dis = nn.Linear(self.encoder_lstm_hidden_size * 2, entity_hidden_size)

        self.sent_represent_dim = self.encoder_hidden_size if use_state else self.encoder_lstm_hidden_size * 2

        self.linear_score = nn.Linear(
            2 * entity_hidden_size + self.sent_represent_dim + distance_embedding_dim, relation_classes
        )

        self.distance_embedding = nn.Embedding(max_distance, distance_embedding_dim)
        self.drop_out = nn.Dropout(drop_out)

        if self.use_ner:
            self.linear_ner_hidden = nn.Linear(self.encoder_lstm_hidden_size * 2, entity_hidden_size)
            self.linear_ner_out = nn.Linear(entity_hidden_size, ner_classes)

    def collect_entity_by_indices(self, representations, positions):

        batch, max_mentions, max_entity_span = positions.shape
        feature_dim = representations.shape[-1]
        positions = positions.view(batch, max_mentions * max_entity_span)
        # positions = [batch, max_mentions * max_entity_span ]
        collected_tensor = torch.gather(
            representations, 1, positions[..., None].expand(*positions.shape, representations.shape[-1])
        )
        collected_tensor = collected_tensor.view(batch, max_mentions, max_entity_span, feature_dim)
        return collected_tensor

    def forward(self, inputs):

        (
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
            distance,
        ) = inputs

        node_cell, node_hidden, lstm_output = self.encoder(
            [
                elmo_tensor,
                token_ids_tensor,
                token_ids_mask_tensor,
                pos_ids_tensor,
                char_ids_tensor,
                in_nodes_idx_tensor,
                in_nodes_mask_tensor,
                out_nodes_idx_tensor,
                out_nodes_mask_tensor,
                in_edge_idx_tensor,
                in_edge_idx_mask,
                out_edge_idx_tensor,
                out_edge_idx_mask,
            ]
        )

        entity_spans = chem_entity_map_tensor.shape[2]
        max_mentions = chem_entity_map_tensor.shape[1]

        sent_represent = torch.max(node_hidden, dim=1)[0]
        sent_represent = sent_represent.unsqueeze(1).unsqueeze(1).repeat(1, max_mentions, max_mentions, 1)

        if self.use_ner:
            ner_hiddens = torch.tanh(self.linear_ner_hidden(self.drop_out(lstm_output)))
            ner_logits = self.linear_ner_out(ner_hiddens)

        rel_representations = self.drop_out(node_hidden)

        collected_chem_entities = self.collect_entity_by_indices(rel_representations, chem_entity_map_tensor)
        collected_chem_entities = collected_chem_entities * chem_entity_map_mask_tensor.unsqueeze(-1)
        chem_entities = torch.sum(collected_chem_entities, dim=2)

        chem_entities = torch.tanh(self.linear_chem(chem_entities))

        collected_dis_entities = self.collect_entity_by_indices(rel_representations, dis_entity_map_tensor)
        collected_dis_entities = collected_dis_entities * dis_entity_map_mask_tensor.unsqueeze(-1)
        dis_entities = torch.sum(collected_dis_entities, dim=2)

        dis_entities = torch.tanh(self.linear_dis(dis_entities))
        distance_embedded = self.distance_embedding(distance)

        # chem_entities = [batch_size, max_mentions, feature_dim]
        # dis_entities = [batch_size, max_mentions, feature_dim]

        bs, max_mentions, feature_dim = chem_entities.shape
        chem_entities_indices_mask = torch.sum(chem_entity_map_mask_tensor, dim=-1).type(torch.cuda.LongTensor)
        dis_entities_indices_mask = torch.sum(dis_entity_map_mask_tensor, dim=-1).type(torch.cuda.LongTensor)

        concat_entities = torch.cat(
            [
                chem_entities.unsqueeze(2).repeat(1, 1, max_mentions, 1),
                dis_entities.unsqueeze(1).repeat(1, max_mentions, 1, 1),
            ],
            dim=-1,
        )
        # concat_entities = [batch_size, max_mentions, max_mentions, feature_dim * 2]

        # print(concat_entities.shape)
        concat_entities = torch.cat([concat_entities, distance_embedded, sent_represent], dim=-1)
        concat_entities = concat_entities.view(
            bs,
            max_mentions * max_mentions,
            2 * self.entity_hidden_size + self.sent_represent_dim + self.distance_embedding_dim,
        )
        # concat_entities = [batch_size, max_mentions * max_mentions, feature_dim * 2]
        # concat_entities = torch.cat([concat_entities, distant_embedded],dim=-1)

        concat_indices_mask = chem_entities_indices_mask.unsqueeze(2).repeat(
            1, 1, max_mentions
        ) * dis_entities_indices_mask.unsqueeze(1).repeat(1, max_mentions, 1)
        # concat_indices_mask = [bs, max_mentions, max_mentions]

        concat_indices_mask = concat_indices_mask.view(bs, max_mentions * max_mentions)
        # concat_indices_mask = [bs, max_mentions * max_mentions]
        distance_mask = (distance >= self.distance_thresh).type(torch.cuda.LongTensor)
        distance_mask = distance_mask.view(bs, max_mentions * max_mentions)
        score = self.linear_score(self.drop_out(concat_entities))
        # score = [bs, max_mentions * max_mentions, 2]

        score = score.masked_fill(concat_indices_mask.unsqueeze(-1) == 0, -1e10)
        score = score.masked_fill(distance_mask.unsqueeze(-1) == 0, -1e10)
        final_score = torch.max(score, dim=1)[0]

        if self.use_ner:
            return ner_logits, final_score

        return final_score
