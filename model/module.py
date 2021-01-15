import torch.nn as nn
import torch

class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        # print(query.shape)
        # print(context.shape)
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)
        # print('query_len', query_len)
        if self.attention_type == "general":
            # print('query 0', query.shape)
            query = query.reshape(batch_size * output_len, dimensions)
            # print('query 1', query.shape)
            query = self.linear_in(query)
            # print('query 2', query.shape)
            query = query.reshape(batch_size, output_len, dimensions)
            # print('query 3', query.shape)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        # print('intent', x.shape)
        # print('intent', self.linear(x).shape)
        return self.linear(x)

class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, num_slot_labels, use_intent_context_concat = False, use_intent_context_attn = False, max_seq_len = 50, intent_embedding_size = 100, attention_embedding_size = 256, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.use_intent_context_attn = use_intent_context_attn
        self.use_intent_context_concat = use_intent_context_concat
        self.max_seq_len = max_seq_len
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.intent_embedding_size = intent_embedding_size
        self.attention_embedding_size = attention_embedding_size

        if self.use_intent_context_concat:
            input_dim = input_dim + self.intent_embedding_size
        
        self.softmax = nn.Softmax(dim = -1)
        
        self.attention = Attention(attention_embedding_size)
        
        if self.use_intent_context_attn:
            self.intent_embedding_size = self.attention_embedding_size

        self.linear_intent_context = nn.Linear(self.num_intent_labels, self.intent_embedding_size, bias = False)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x, intent_context):
        if self.use_intent_context_concat:
            intent_context = self.softmax(intent_context)
            intent_context = self.linear_intent_context(intent_context)
            intent_context = torch.unsqueeze(intent_context, 1)
            intent_context = intent_context.expand(-1, self.max_seq_len, -1)
            hidden_size = x.shape[2]
            x = nn.ConstantPad1d((0,self.intent_embedding_size), 1)(x)
            x[:,:,hidden_size:] = intent_context
        
        elif self.use_intent_context_attn:
            intent_context = self.softmax(intent_context)
            intent_context = self.linear_intent_context(intent_context)
            intent_context = torch.unsqueeze(intent_context, 1) #1: query length (each token)
            # intent_context = intent_context.expand(-1, self.max_seq_len, -1)
            output, weights = self.attention(x, intent_context)
            x = output
        x = self.dropout(x)
        return self.linear(x)