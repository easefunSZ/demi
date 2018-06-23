''''
--- Handles the data used for neural net training and validation.
-- @classmod data_stream
'''''

# require 'torch'
arguments = require
'Settings.arguments'


class DataStream(object):
    '''
    --- Constructor.
    --
    -- Reads the data from training and validation files generated with
    -- @{data_generation_call.generate_data}.
    '''

    def __init(self):
        ##--loadind valid data
        self.data = {}
        valid_prefix = arguments.data_path + 'valid'
        self.data.valid_mask = torch.load(valid_prefix + '.mask')
        self.data.valid_mask = self.data.valid_mask.repeatTensor(1, 2)
        self.data.valid_targets = torch.load(valid_prefix + '.targets')
        self.data.valid_inputs = torch.load(valid_prefix + '.inputs')
        self.valid_data_count = self.data.valid_inputs.size(1)
        assert (self.valid_data_count >= arguments.train_batch_size,
                'Validation data count has to be greater than a train batch size!')
        self.valid_batch_count = self.valid_data_count / arguments.train_batch_size
        ##--loading train data
        train_prefix = arguments.data_path + 'train'
        self.data.train_mask = torch.load(train_prefix + '.mask')
        self.data.train_mask = self.data.train_mask.repeatTensor(1, 2)
        self.data.train_inputs = torch.load(train_prefix + '.inputs')
        self.data.train_targets = torch.load(train_prefix + '.targets')
        self.train_data_count = self.data.train_inputs.size(1)
        assert (self.train_data_count >= arguments.train_batch_size,
                'Training data count has to be greater than a train batch size!')
        self.train_batch_count = self.train_data_count / arguments.train_batch_size

        ##--transfering data to gpu if needed
        if arguments.gpu:
            for key, value in pairs(self.data):
                self.data[key] = value.cuda()

    '''
    --- Gives the number of batches of validation data.
    --
    -- Batch size is defined by @{arguments.train_batch_size}.
    -- @return the number of batches
    '''

    def get_valid_batch_count(self):
        return self.valid_batch_count

    '''
    --- Gives the number of batches of training data.
    --
    -- Batch size is defined by @{arguments.train_batch_size}
    -- @return the number of batches
    '''

    def get_train_batch_count(self):
        return self.train_batch_count

    '''
    --- Randomizes the order of training data.
    --
    -- Done so that the data is encountered in a different order for each epoch.
    '''

    def start_epoch(self):
        ##--data are shuffled each epoch
        shuffle = torch.randperm(self.train_data_count).long()

        self.data.train_inputs = self.data.train_inputs.index(1, shuffle)
        self.data.train_targets = self.data.train_targets.index(1, shuffle)
        self.data.train_mask = self.data.train_mask.index(1, shuffle)

    '''
    --- Returns a batch of data from a specified data set.
    -- @param inputs the inputs set for the given data set
    -- @param targets the targets set for the given data set
    -- @param mask the masks set for the given data set
    -- @param batch_index the index of the batch to return
    -- @return the inputs set for the batch
    -- @return the targets set for the batch
    -- @return the masks set for the batch
    -- @local
    '''

    def get_batch(self, inputs, targets, mask, batch_index):
        assert (inputs.size(1) == targets.size(1) and inputs.size(1) == mask.size(1))
        batch_boundaries = {(batch_index - 1) * arguments.train_batch_size + 1,
                            batch_index * arguments.train_batch_size}
        batch_table_index = {batch_boundaries, {}}
        batch_inputs = inputs[batch_table_index]
        batch_targets = targets[batch_table_index]
        batch_mask = mask[batch_table_index]
        return batch_inputs, batch_targets, batch_mask

    '''
    --- Returns a batch of data from the training set.
    -- @param batch_index the index of the batch to return
    -- @return the inputs set for the batch
    -- @return the targets set for the batch
    -- @return the masks set for the batch
    '''

    def get_train_batch(self, batch_index):
        return self.get_batch(self.data.train_inputs, self.data.train_targets, self.data.train_mask, batch_index)

    '''
    --- Returns a batch of data from the validation set.
    -- @param batch_index the index of the batch to return
    -- @return the inputs set for the batch
    -- @return the targets set for the batch
    -- @return the masks set for the batch
    '''

    def get_valid_batch(self, batch_index):
        return self.get_batch(self.data.valid_inputs, self.data.valid_targets, self.data.valid_mask, batch_index)
