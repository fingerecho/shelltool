var program = require('commander');

var I64BIT_TABLE =
 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-'.split('');
function exec(flag){
 var input = flag;
 var hash = 5381;
 var i = input.length - 1;
 if(typeof input == 'string'){
  for (; i > -1; i--)
   hash += (hash << 5) + input.charCodeAt(i);
 }
 else{
  for (; i > -1; i--)
   hash += (hash << 5) + input[i];
 }
 var value = hash & 0x7FFFFFFF;
  
 var retValue = '';
 do{
  retValue += I64BIT_TABLE[value & 0x3F];
 }
 while(value >>= 6);  
 //return retValue;
 console.log(retValue);
}

program
    .command('exec <flag>')
    .action(function(flag){
        exec(flag);
    });
program.parse(process.argv);