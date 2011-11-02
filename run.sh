#!/bin/bash

time bin/terminator 0 ~/Corpus/trec05p/full/ trec05p.0.result
time bin/terminator 0 ~/Corpus/trec06p/full/ trec06p.0.result
time bin/terminator 0 ~/Corpus/trec06c/full/ trec06c.0.result
time bin/terminator 0 ~/Corpus/trec07p/full/ trec07p.0.result
time bin/terminator 0 ~/Corpus/ceas08/full-immediate/ ceas08.0.result
time bin/terminator 0 ~/Corpus/sewm08/full/ sewm08.0.result
time bin/terminator 0 ~/Corpus/sewm10/full/ sewm10.0.result

time bin/terminator 5 ~/Corpus/trec05p/full/ trec05p.5.result
time bin/terminator 5 ~/Corpus/trec06p/full/ trec06p.5.result
time bin/terminator 5 ~/Corpus/trec06c/full/ trec06c.5.result
time bin/terminator 5 ~/Corpus/trec07p/full/ trec07p.5.result
time bin/terminator 5 ~/Corpus/ceas08/full-immediate/ ceas08.5.result
time bin/terminator 5 ~/Corpus/sewm08/full/ sewm08.5.result
time bin/terminator 5 ~/Corpus/sewm10/full/ sewm10.5.result
