import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SelectNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class QueryCentricNet(SelectNet):
    def __init__(self, embedding, max_match=20, win_size=15, dim_w=2, dim_v=2):
        super().__init__()

        self.max_match = max_match
        self.win_size = win_size

        if type(embedding) is np.ndarray:
            embedding = torch.from_mumpy(embedding)
        self.embedding = nn.Embedding.from_pretrained(embedding)

        self.fc_w = nn.Linear(self.embedding.embedding_dim, dim_w)
        self.fc_v = nn.Linear(self.embedding.embedding_dim, dim_v)

    def get_win(self, document, position):
        start_id = max(0, position-self.win_size)
        stop_id = min(len(document)-1, start_id+2*)


    def forward(self, query, document):
        inputs = []
        queries = []
        for w_id in query:
            queries.append(self.embedding(w_id))
            matches = []
            for v_id in document:
                if len(matches) > self.max_match:
                    break
                if v_id == w_id:
                    matches.append(self.embedding(w_id))
            inputs.append(matches)
        query = torch.cat(queries, dim=1)
        return input_tensor


class PointerNet(SelectNet):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

