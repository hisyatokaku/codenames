## spymaster strategy

![eq1](https://latex.codecogs.com/gif.latex?clue%20%3D%20%5Cunderset%7Bword%20%5Cin%20%7CV%7C%7D%7Bargmax%7D%20%28%7B%5Csum_%7Bcard_i%20%5Cin%20Pos%20%5Csubset%7BPos_%7Ball%7D%7D%7D%7Bcos%28card_i%2C%20word%29%7D%7D%29)
$Pos$ is uniquely picked by following constraints.
![eq2](https://latex.codecogs.com/gif.latex?%5C%5C%20%5C%7BPos%20%7C%20%28%5Cforall%7Bcard_p%20%5Cin%20Pos%7D%2C%20%5Cforall%7Bcard_n%20%5Cin%20Neg%7D%29%2C%20cos%28card_p%2C%20word%29%20%5Cgt%20max%28cos%28card_n%2C%20word%29%29%5C%7D)

## player strategy
add gaussian noise vector to pretrained embeddings.
noise parameters are given by the forms of standard deviation.
For example, the last part of the directory name, std0.2, means that the experiment was done in
the setting of std = 0.2 .