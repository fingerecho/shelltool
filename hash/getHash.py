I64BIT_TABLE ='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-'.split();
I64BIT_TABLE_str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-'

def hash(input):
	hsh_ = 5381;
	len_i = len(input) -1;
	if isinstance(input,str):
		for i in range(len_i,-1,-1):
			hsh_ += (hsh_<<5) 
			hsh_ += ord(input[i]);
	else:
		for i in range(len_i,-1,-1):
			hsh_ += (hsh_ << 5) + input[i];
	value = hsh_ & 0x7FFFFFFF;
	#print(value&0x3F)
	retValue = '';
	retValue += I64BIT_TABLE_str[value&0x3F];
	while value > 0:
		retValue += I64BIT_TABLE_str[value&0x3F];
		value = value >> 6
	return retValue[1:]


