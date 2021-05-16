with open('category_mapping.txt', 'r') as file:
   lines = file.readlines()
   split_lines = [line.split(' ') for line in lines]
   
with open('category_mapping_fixed.txt', 'w') as file:
   # shift up by one with unknown classification element
   split_lines.insert(25, ['25', '?\n'])
   for i in range(25,len(split_lines)):
      split_lines[i][0] = int(split_lines[i][0]) + 1
   for line in split_lines:
      file.write(f'{line[0]} {line[1]}')
   