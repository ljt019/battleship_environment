# battleship_environment

Just a fun learning project, wanted to try training a LLM. I chose Battleship because it seemed like a fun way to try out rlvr, and figuring out how to represent the grid-like board state to the llm seems like a fun challenge.

Checkout [verifiers](https://github.com/willccbb/verifiers) it's awesome!

## Sources I found helpful 

#### [A Llama Sunk My Battleship!](https://openreview.net/pdf/75c9c75047025b188ec2a084d248e8662e2e3a4e.pdf) 

Interesting paper that uses in context examples to try and teach gpt-4 to play battleship, particularly useful for me in that they tried three different methods of representing the board to the model:

- Grid: classic grid style (what i plan to use)
- Textual: text representation (i.e. 1-C is a water tile)
- Visual: image of the board state

The papers results showed textual representations performed the best, I'm hoping to explore grid based representations more though, to see if modern models perform the same, an important difference in my case is I'm actually training the model on the style, which may have a diffrent outcome, we'll see! I mainly chose the grid representation because it's the one I find most interesting.
