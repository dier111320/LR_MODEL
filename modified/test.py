
def file_append(source_filename,dest_file_name):
    source_f = open(source_filename,'r')
    dest_f = open(dest_file_name,'a')
    dest_f.write('\n')
    dest_f.write(source_f.read())
    source_f.close()
    dest_f.close()

file_append('test2.txt', 'test1.txt')