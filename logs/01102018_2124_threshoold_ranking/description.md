## spymaster strategy
ranking function 
$$ score(word) = \underset{w \in |V|}{argmax}
(
{\sum_{i \in Pos \subset{Pos_{all}}}{\frac {cos(card_i, word)} {|Pos|} } - \sum_{j\in Neg}{\frac {cos(card_j, word)} {|Neg|} }} ) ) $$

## player strtegy


