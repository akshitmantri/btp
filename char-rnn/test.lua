--t = "I am a very good boy.\nI like to play fifa."

f = assert(io.open('../data/char-rnn/input.txt', 'r'))
t = f:read("*all")
f:close()

t = string.gsub(t,"\n"," ")
_, count = string.gsub(t, " ", "")
dataset = torch.IntTensor(count+1)
data_index = 1

--print(p)

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

s = 0
for u,v in pairs(dictionary) do
	print (u,v)
	s = s+1
end
print(dataset)
print (s)

