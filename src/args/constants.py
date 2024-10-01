## Useful Constants for the Analysis of the fMRI measurements
Naturals    = [[5, 165, 'Natural'],  [9, 119, 'Natural']]
Artificials = [[3, -41, 'Artificial'], [1, -212, 'Artificial']]
Backgrounds = Naturals + Artificials; nBackgrounds = len(Backgrounds)

Animals = [['Elephant', 'Animal'], ['Toucan', 'Animal']]
Tools   = [['Iron', 'Tool'], ['Teapot', 'Tool']]
Objects = Animals + Tools; nObjects = len(Objects)

Numerosity = [1, 2, 3, 4]; nNumerosity = len(Numerosity)
nConditions = nNumerosity*nObjects*nBackgrounds