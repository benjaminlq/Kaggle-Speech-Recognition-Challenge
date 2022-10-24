import torch

class BeamSearchNode(object):
    def __init__(self, hidden, previousNode, wordId, logProb, length, cell = None):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hidden
        self.c = cell
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward  # 注意这里是有惩罚参数的，参考恩达的 beam-search

    def __lt__(self, other):
        return self.leng < other.leng  

    def __gt__(self, other):
        return self.leng > other.leng

def beam_decode(model, proc, out_len, encoder_hiddens, hidden, cell):
    '''
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = proc.get_model_arg('beam_width')
    topk = proc.get_model_arg('beam_topk')  # how many sentence do you want to generate

    if beam_width is None:
        beam_width = 10
    if topk is None:
        topk = 1
    # Start with the start of the sentence token
    decoder_input = torch.tensor([proc.beg_token_id], dtype=torch.long).to(device)
    
    # Number of sentence to generate
    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))

    # starting node -  hidden vector, previous node, word id, logp, length
    node = BeamSearchNode(hidden, cell, None, decoder_input, 0, 1)
    nodes = PriorityQueue()

    # start the queue
    nodes.put((-node.eval(), node))
    qsize = 1

    # start beam search
    while True:
        # give up when decoding takes too long
        if qsize > 2000: break

        # fetch the best node
        score, n = nodes.get()
        decoder_input = n.wordid
        decoder_hidden = n.h

        if n.wordid.item() == proc.end_token_id and n.prevNode != None:
            endnodes.append((score, n))
            # if we reached maximum # of sentences required
            if len(endnodes) >= number_required:
                break
            else:
                continue

        # decode for one step using decoder
        output, hidden, cell = model.decoder(decoder_input, encoder_hiddens, hidden, cell)
        # output.shape: 1 * vocab_size
        # hidden.shape: n_layer * 1 * vocab_size
        # cell.shape: n_layer * 1 * vocab_size
        # PUT HERE REAL BEAM SEARCH OF TOP
        log_prob, indexes = torch.topk(output, beam_width)
        nextnodes = []
        # log_prob.shape: 1 * beam_width
        # indexes.shape: 1 * beam_width
        for new_k in range(beam_width):
            decoded_t = indexes[0][new_k].view(-1)
            log_p = log_prob[0][new_k].item()
            node = BeamSearchNode(hidden, cell, n, decoded_t, n.logp + log_p, n.leng + 1)
            score = -node.eval()
            nextnodes.append((score, node))

        # put them into queue
        for i in range(len(nextnodes)):
            score, nn = nextnodes[i]
            nodes.put((score, nn))
            # increase qsize
        qsize += len(nextnodes) - 1

    # choose nbest paths, back trace them
    if len(endnodes) == 0:
        endnodes = [nodes.get() for _ in range(topk)]

    utterances = []
    for score, n in sorted(endnodes, key=operator.itemgetter(0)):
        utterance = []
        utterance.append(n.wordid.item())
        # back trace
        while n.prevNode != None:
            n = n.prevNode
            utterance.append(n.wordid.item())

        utterance = utterance[::-1]
        utterances.append(utterance[:out_len])

    return utterances

def beam_translate(model, proc, out_len, input_sent, device):
    model.eval() # eval mode
    tokens = en_preprocess(input_sent)
    input_seq = [proc.beg_token_id] + [proc.lang1.w2id_vocab.get(t, proc.unk_token_id) for t in tokens] + [proc.end_token_id]
    input_ids = torch.tensor(input_seq, dtype=torch.long).unsqueeze(1)
    encoder_hiddens, hidden, cell = model.encoder(input_ids.to(device))

    decoded_batch = beam_decode(model, proc, out_len, encoder_hiddens, hidden, cell)

    return [' '.join([proc.lang2.id2w_vocab[tid] for tid in sent]) for sent in decoded_batch]

def greedy_translate(model, proc, out_len, input_sent, device):
    model.eval() # eval mode
    tokens = en_preprocess(input_sent)
    input_seq = [proc.beg_token_id] + [proc.lang1.w2id_vocab.get(t, proc.unk_token_id) for t in tokens] + [proc.end_token_id]
    input_ids = torch.tensor(input_seq, dtype=torch.long).unsqueeze(1)
    encoder_hiddens, hidden, cell = model.encoder(input_ids.to(device))
    
    #first input to the decoder is the <cls> tokens
    outputs = []
    trg_seq = torch.tensor([proc.beg_token_id], dtype=torch.long)
    
    for t in range(1, out_len):
        #insert input token embedding, previous hidden and previous cell states
        #receive output tensor (predictions) and new hidden and cell states
        output, hidden, cell = model.decoder(trg_seq.to(device), encoder_hiddens, hidden, cell)
        trg_seq = output.argmax(1)
        if (trg_seq.item() == proc.end_token_id): break
        outputs.append(trg_seq.item())

    return ' '.join([proc.lang2.id2w_vocab[tid] for tid in outputs])

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.1, 0.1)