--preprocess the data, build a dictionary of all the words in the inputfile. and also build the dataset with words replaced 
-- with indices in the dictionary

function process_file(input_file)

	f = assert(io.open(input_file, 'r'))
	t = f:read("*all")
	f:close()

	t = string.gsub(t,"\n"," ")
	_, count = string.gsub(t, " ", "")
	dataset = torch.IntTensor(count+1)
	data_index = 1
	dictionary = {}
	cur_index = 1
	prev=1
	num_words = 0
	for i=1,#t do
		if (string.sub(t,i,i) == ' ') then
			if not dictionary[string.sub(t,prev,i-1)] then 
				dictionary[string.sub(t,prev,i-1)] = cur_index
				cur_index = cur_index+1
			end
			dataset[data_index] = dictionary[string.sub(t,prev,i-1)]
			data_index = data_index+1
			prev = i+1
		end
	end
	if not dictionary[string.sub(t,prev)] then
		dictionary[string.sub(t,prev)] = cur_index
	end
	dataset[data_index] = dictionary[string.sub(t,prev)]

	return dataset,dictionary
end

dataset, dictionary = process_file('../data/char-rnn/input.txt')
torch.save('../data/char-rnn/input_words.t7', dataset)
torch.save('../data/char-rnn/dictionary_words.t7', dictionary)
