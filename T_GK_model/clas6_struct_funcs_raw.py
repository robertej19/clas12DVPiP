# from https://arxiv.org/pdf/1405.0988.pdf

import pandas as pd

# Input data string
#columns:
#Q2, xB, t, tel,tel_stat,tel_sys, lt,  lt_stat, lt_sys, tt, tt_stat, tt_sys
data = """
1.14 0.131 0.12 341 ± 40 ± 59 −30 ± 68 ± 114 −240 ± 111 ± 156
1.15 0.132 0.17 314 ± 40 ± 75 −76 ± 69 ± 126 −292 ± 108 ± 215
1.15 0.132 0.25 267 ± 19 ± 15 −42 ± 32 ± 37 −233 ± 55 ± 21
1.15 0.132 0.35 188 ± 13 ± 33 −50 ± 23 ± 43 −179 ± 43 ± 66
1.15 0.132 0.49 126.3 ± 4.7 ± 10 −15.0 ± 8.0 ± 5.5 −78 ± 19 ± 8.1
1.15 0.132 0.77 66.0 ± 2.0 ± 7.9 3.8 ± 3.1 ± 6.4 −39.8 ± 7.8 ± 16
1.16 0.133 1.71 17.8 ± 2.0 ± 1.6 4.3 ± 1.2 ± 2.0 −21.2 ± 6.6 ± 7.7
1.38 0.169 0.12 357 ± 13 ± 35 19 ± 19 ± 30 −191 ± 42 ± 47
1.38 0.169 0.17 366 ± 15 ± 24 2 ± 22 ± 21 −247 ± 46 ± 53
1.38 0.169 0.25 331 ± 12 ± 16 19 ± 18 ± 17 −202 ± 36 ± 49
1.38 0.169 0.35 254 ± 10 ± 13 17 ± 15 ± 24 −153 ± 32 ± 25
1.38 0.169 0.49 166.2 ± 5.1 ± 12 −15.4 ± 7.1 ± 12 −109 ± 18 ± 18
1.38 0.169 0.77 83.4 ± 3.3 ± 4.1 9.7 ± 4.4 ± 10 −48.5 ± 9.6 ± 5.4
1.38 0.169 1.21 39.6 ± 1.7 ± 3.8 4.0 ± 1.7 ± 1.9 −40.8 ± 4.5 ± 3.0
1.38 0.170 1.71 15.3 ± 1.4 ± 1.5 0.81 ± 0.80 ± 1.6 −13.6 ± 4.0 ± 5.1
1.61 0.186 0.12 276 ± 17 ± 46 17 ± 29 ± 58 −180 ± 64 ± 71
1.61 0.186 0.18 345 ± 25 ± 57 36 ± 42 ± 102 −103 ± 82 ± 87
1.61 0.187 0.25 276 ± 15 ± 7.0 0 ± 26 ± 21 −171 ± 52 ± 41
1.61 0.187 0.35 223 ± 12 ± 11 −14 ± 20 ± 11 −143 ± 46 ± 46
1.61 0.187 0.49 159.8 ± 6.3 ± 11 20 ± 10 ± 11 −58 ± 25 ± 19
1.61 0.187 0.78 82.4 ± 3.2 ± 7.1 5.6 ± 4.8 ± 19 −30 ± 12 ± 27
1.61 0.187 1.21 34.5 ± 2.3 ± 3.0 0.1 ± 3.3 ± 1.7 −24.9 ± 6.4 ± 6.6
1.61 0.187 1.71 16.0 ± 1.9 ± 1.6 2.3 ± 1.8 ± 2.2 −12.2 ± 6.2 ± 4.6
1.74 0.223 0.25 316.7 ± 6.7 ± 9.2 14.9 ± 8.5 ± 19 −232 ± 20 ± 44
1.75 0.223 0.12 293.3 ± 7.8 ± 24 16.2 ± 9.8 ± 12 −72 ± 23 ± 13
1.75 0.223 0.17 339.3 ± 8.9 ± 26 35 ± 11 ± 8.3 −243 ± 28 ± 26
1.75 0.224 0.35 260.5 ± 7.0 ± 13 32.1 ± 9.2 ± 5.0 −183 ± 22 ± 20
1.75 0.224 0.49 184.4 ± 5.0 ± 8.6 3.6 ± 6.3 ± 3.7 −116 ± 15 ± 20
1.75 0.224 0.78 102.2 ± 2.4 ± 5.4 9.2 ± 3.1 ± 5.0 −61.0 ± 7.3 ± 12
1.75 0.224 1.22 44.5 ± 1.4 ± 3.0 6.3 ± 1.3 ± 2.2 −21.2 ± 4.1 ± 6.0
1.75 0.224 1.72 19.00 ± 1.00 ± 4.4 2.24 ± 0.85 ± 3.2 −12.3 ± 3.0 ± 5.4
1.87 0.270 0.12 342 ± 74 ± 108 1 ± 86 ± 72 −150 ± 103 ± 101
1.87 0.271 0.18 437 ± 54 ± 90 7 ± 64 ± 74 16 ± 91 ± 167
1.87 0.271 0.25 412 ± 19 ± 32 20 ± 21 ± 20 −233 ± 34 ± 39
1.87 0.271 0.35 374 ± 14 ± 26 27 ± 13 ± 20 −293 ± 28 ± 41
1.87 0.271 0.49 259.5 ± 7.3 ± 13 25.1 ± 7.2 ± 6.1 −167 ± 19 ± 14
1.87 0.271 0.78 151.8 ± 4.1 ± 7.8 6.4 ± 4.2 ± 5.7 −59 ± 12 ± 4.6
1.87 0.271 1.22 77.7 ± 3.0 ± 5.5 −5.7 ± 2.3 ± 2.8 −36.4 ± 7.4 ± 5.6
1.87 0.272 1.72 39.2 ± 2.1 ± 3.5 −7.0 ± 1.9 ± 1.9 −22.9 ± 4.6 ± 3.8
1.95 0.313 0.35 470 ± 44 ± 82 −13 ± 34 ± 18 −183 ± 77 ± 58
1.95 0.313 0.49 339 ± 23 ± 21 21 ± 15 ± 34 −140 ± 50 ± 43
1.95 0.313 0.78 202 ± 12 ± 13 −11.1 ± 9.4 ± 5.8 −67 ± 31 ± 23
1.96 0.313 1.22 129.4 ± 9.6 ± 17 −24.8 ± 8.3 ± 6.7 −39 ± 22 ± 21
2.10 0.238 0.12 258 ± 33 ± 81 79 ± 51 ± 109 179 ± 126 ± 218
2.10 0.238 0.35 219 ± 18 ± 8.1 95 ± 31 ± 10 91 ± 72 ± 46
2.10 0.238 0.49 132.5 ± 8.9 ± 13 −53 ± 15 ± 9.0 −105 ± 41 ± 28
2.10 0.238 0.78 92.6 ± 8.9 ± 9.2 −8 ± 13 ± 12 21 ± 35 ± 32
2.10 0.238 1.21 40 ± 21 ± 16 −6 ± 35 ± 31 −23 ± 43 ± 27
2.10 0.239 0.17 228 ± 29 ± 148 −13 ± 49 ± 265 −7 ± 119 ± 268
2.10 0.239 0.25 240 ± 20 ± 24 57 ± 36 ± 30 47 ± 83 ± 106
2.21 0.275 0.12 241 ± 25 ± 11 −44 ± 36 ± 9.0 29 ± 58 ± 17
2.21 0.276 0.17 257 ± 12 ± 18 −6 ± 17 ± 13 −13 ± 38 ± 41
2.21 0.276 0.25 268.8 ± 9.8 ± 19 −6 ± 13 ± 20 −54 ± 29 ± 30
2.21 0.276 0.35 242 ± 11 ± 11 32 ± 14 ± 12 −102 ± 34 ± 22
2.21 0.276 0.49 193.5 ± 7.1 ± 17 41.1 ± 9.4 ± 20 −56 ± 22 ± 47
2.21 0.276 0.78 101.4 ± 3.0 ± 6.6 7.3 ± 4.3 ± 7.0 −69 ± 10 ± 10
2.21 0.277 1.22 50.0 ± 2.0 ± 3.3 5.8 ± 2.3 ± 3.9 −22.5 ± 6.9 ± 2.4
2.21 0.277 1.72 20.8 ± 1.5 ± 3.1 −0.1 ± 1.8 ± 2.3 −10.1 ± 4.8 ± 5.3
2.24 0.332 0.18 330 ± 44 ± 31 14 ± 53 ± 37 −114 ± 80 ± 118
2.24 0.337 0.25 392 ± 19 ± 44 −8 ± 20 ± 34 −53 ± 34 ± 27
2.24 0.338 0.49 293.7 ± 6.5 ± 15 26.4 ± 5.5 ± 13 −137 ± 14 ± 12
2.25 0.337 0.35 346 ± 12 ± 14 40 ± 11 ± 12 −152 ± 24 ± 15
2.25 0.338 0.78 200.8 ± 3.8 ± 13 −2.1 ± 3.3 ± 5.0 −78.6 ± 9.7 ± 10
2.25 0.339 1.22 110.2 ± 2.6 ± 5.4 −13.3 ± 2.3 ± 4.2 −50.4 ± 6.5 ± 6.1
2.25 0.339 1.73 49.9 ± 1.7 ± 4.6 −6.5 ± 1.8 ± 5.7 −32.3 ± 3.7 ± 5.8
2.34 0.403 0.35 472 ± 48 ± 53 −6 ± 60 ± 79 −24 ± 105 ± 210
2.34 0.403 0.49 475 ± 20 ± 39 −22 ± 23 ± 27 −17 ± 51 ± 53
2.34 0.404 0.78 377 ± 11 ± 17 −22 ± 10 ± 5.8 −150 ± 26 ± 19
2.34 0.404 1.22 192.8 ± 7.4 ± 13 −37.3 ± 7.9 ± 4.4 −67 ± 16 ± 43
2.35 0.404 1.73 90.5 ± 6.6 ± 3.1 −22.4 ± 7.4 ± 5.7 −13 ± 12 ± 8.4
2.71 0.336 0.18 230 ± 35 ± 29 −78 ± 52 ± 84 60 ± 90 ± 188
2.71 0.343 0.25 217.3 ± 8.1 ± 10 −6 ± 10 ± 4.3 −76 ± 27 ± 22
2.71 0.343 0.35 220.5 ± 8.1 ± 8.0 15.5 ± 9.8 ± 7.6 −97 ± 27 ± 28
2.71 0.343 0.49 183.8 ± 6.0 ± 9.4 17.0 ± 7.4 ± 12 −120 ± 19 ± 31
2.71 0.343 1.22 51.3 ± 2.4 ± 4.5 9.0 ± 2.7 ± 5.0 −31.5 ± 9.7 ± 16
2.72 0.344 0.78 110.4 ± 3.6 ± 5.8 1.8 ± 4.7 ± 5.8 −99 ± 14 ± 20
2.72 0.344 1.73 28.7 ± 1.9 ± 3.5 −2.9 ± 2.2 ± 2.0 −17.2 ± 5.6 ± 9.2
2.75 0.423 0.50 323 ± 19 ± 21 −8 ± 23 ± 16 −60 ± 40 ± 16
2.75 0.423 0.78 232.4 ± 6.9 ± 17 4.3 ± 6.4 ± 16 −58 ± 17 ± 24
2.75 0.424 1.23 140.7 ± 4.9 ± 9.0 −25.8 ± 5.6 ± 5.8 −16 ± 13 ± 12
2.75 0.424 1.73 69.3 ± 4.6 ± 2.9 −12.8 ± 5.3 ± 3.7 −2.7 ± 9.6 ± 12
3.12 0.362 0.35 219 ± 33 ± 139 1 ± 53 ± 213 27 ± 114 ± 398
3.12 0.362 0.50 167 ± 14 ± 20 1 ± 23 ± 59 −21 ± 71 ± 56
3.22 0.431 0.78 138.4 ± 6.2 ± 6.5 15.0 ± 7.9 ± 5.5 −77 ± 17 ± 16
3.23 0.428 0.35 277 ± 22 ± 15 −80 ± 29 ± 16 67 ± 48 ± 20
3.23 0.430 0.50 201 ± 12 ± 17 10 ± 16 ± 17 −46 ± 30 ± 31
3.23 0.432 1.23 75.5 ± 3.8 ± 9.2 5.6 ± 4.3 ± 12 −77 ± 11 ± 32
3.23 0.432 1.73 65.4 ± 5.0 ± 6.7 18.8 ± 5.7 ± 6.2 35 ± 14 ± 15
3.29 0.496 1.23 140 ± 17 ± 18 −12 ± 23 ± 9.7 −54 ± 45 ± 12
3.67 0.451 0.78 145 ± 36 ± 23 −22 ± 35 ± 28 8 ± 101 ± 56
3.67 0.451 1.23 77 ± 15 ± 1.8 2 ± 17 ± 2.9 −24 ± 48 ± 8.8
3.68 0.451 0.49 185 ± 26 ± 18 −32 ± 39 ± 29 −38 ± 66 ± 57
3.68 0.451 1.73 47.0 ± 6.9 ± 3.9 −14.7 ± 9.4 ± 7.3 −27 ± 27 ± 7.9
3.76 0.513 0.78 190 ± 37 ± 40 24 ± 46 ± 37 −39 ± 56 ± 41
3.76 0.514 1.23 132 ± 13 ± 11 1 ± 14 ± 8.4 −17 ± 37 ± 40
4.23 0.539 0.78 178 ± 42 ± 45 −28 ± 60 ± 57 −34 ± 74 ± 64
"""
import sys

# Preprocessing
lines = data.strip().split("\n")
split_lines = [line.split() for line in lines]

# Splitting each line into measurements and errors
processed_lines = []
for line in split_lines:
    print(line)
    # remove the plus/minus sign
    line = [item.replace("±", "") for item in line]
    #replace double spaces with single spacesz
    line = [item.replace("   ", " ") for item in line]
    line = [item.replace("  ", " ") for item in line]
    #remove any empty strings
    line = [item for item in line if item != ""]

    #replace spaces with commas

    line = [item.replace(" ", ",") for item in line]


    print(line)

    # processed_line = []
    # for item in line:
    #     # Checking if an item contains error information
    #     if "±" in item:
    #         values = item.split("±")
    #         # Adding individual values to the line
    #         processed_line.extend(values)
    #     else:
    #         processed_line.append(item)
    processed_lines.append(line)

# Defining column labels
column_labels = ['Q2_C6', 'xB_C6', 't_C6', 'tel_C6', 'telstat_C6', 'telsys_C6', 'lt_C6', 'ltstat_C6', 'ltsys_C6', 'tt_C6', 'ttstat_C6', 'ttsys_C6']

print(processed_lines)
# Creating DataFrame
df = pd.DataFrame(processed_lines, columns=column_labels)

# Saving DataFrame to CSV
#df.to_csv('output.csv', index=False)
print(df)
df.to_pickle("CLAS6_struct_funcs_raw.pkl")