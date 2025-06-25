%
% This script classify the textures of the dataset (AIPI P3 Texture Classification). 
%
% The first loop is used to extract the features of:
%  - the training set (first 40 images of each class)
%  - the testing set (rest of images of each class: 40 per class)
%
% Note that, it loads all the images (they should be in the 'P3_class/' folder!), 
% extract the features using the function computeFeatureVector and save the
% features in matrices:
%
%    vecTrain: contains the training features for each image
%    labTrain: contains the class labels (1..28) for each training image
%    vecTest: contains the features for each of the testing images
%    labTest: contains the class labels (1..28) for each testing image
%
% With these matrices you have to train your K-NN (Nearest Neighbour) Classifier 
%
% You should compute the confusion matrix to obtain the results and also the % of 
% correct classification (accuracy). For a perfect classifier, the numbers of the 
% confusion matrix should be only in the diagonal. From the matrix you can compute
% the % of correct classification = sum of the diagonal / sum of the confusion matrix.


close all;
clear all;

dataDir = 'P3_class/';
d = dir([dataDir 't*']);

nTrain = 40; %number of training images
nTest = 40; %number of testing images

%computing features from training and testing sets
for i=1:length(d)
	namedir = d(i).name;
	d1 = dir([dataDir namedir '/*.jpg']);
	for j=1:nTrain
		name = [dataDir namedir '/' d1(j).name];
		A = imread(name);
		vecTrain((i-1)*nTrain+j,:) = computeFeatureVector(A); %training vector
		labTrain((i-1)*nTrain+j) = i; %training labels
	end
	for j=nTrain+1:nTrain+nTest
		name = [dataDir namedir '/' d1(j).name];
		A = imread(name);
		vecTest((i-1)*nTest+j-nTrain,:) = computeFeatureVector(A); %testing vector
		labTest((i-1)*nTest+j-nTrain) = i; %testing labels
	end
end


%--------------------IMPLEMENTACIÓ----------------------



% --- CLASSIFICADOR K-NN ---

k = 1;  % Nombre de veïns (K) a considerar. En aquest cas, només el més proper

% Entrenem un model K-NN amb les dades d'entrenament
% fitcknn crea un classificador basat en els veïns més propers
% 'Standardize',1 normalitza les dades (evita que les característiques amb valors grans dominin)
Mdl = fitcknn(vecTrain, labTrain, 'NumNeighbors', k, 'Standardize', 1);

% Fem la predicció sobre les dades de test
% predict aplica el model entrenat als vectors de test
vec = predict(Mdl, vecTest);


% --- CÀLCUL DE LA MATRIU DE CONFUSIÓ I L'EXACTITUD ---

% Creem la matriu de confusió comparant les etiquetes reals amb les prediccions
% Cada fila representa la classe real, i cada columna la classe predita
c = confusionmat(labTest, vec);

% Mostrem la matriu de confusió en format visual (gràfic)
confusionchart(c)

% Calculem l'exactitud (accuracy)
% La suma de la diagonal principal indica el nombre de classificacions correctes
% La suma total de la matriu és el nombre total d'exemples
accuracy = sum(diag(c)) / sum(c(:))




 

