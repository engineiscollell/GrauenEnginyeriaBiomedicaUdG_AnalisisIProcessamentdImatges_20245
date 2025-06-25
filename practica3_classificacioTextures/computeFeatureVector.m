function v = computeFeatureVector(A)
%
% Describe an image A using a feature vector.
%   A is the image
%   v is a 1xN vector, being N the number of features used to describe the
% image. 

% Example that returns as a feature the mean intensity of the input image A
% You should modify this code in order compute the GLCM features!! Note
% that the output should be a vector with the features!! contrast,
% homogeneity, energy, etc computed at diferent angles and distances.


%----------------IMPLEMENTACIÓ-----------


% Si la imatge és en color (RGB), la convertim a escala de grisos
if size(A,3) == 3
	A = rgb2gray(A);
end

% Ens assegurem que la imatge estigui en format uint8 (8 bits per píxel),
% ja que algunes funcions com graycomatrix ho requereixen
A = uint8(A);

% Definim els desplaçaments (offsets) per construir la matriu de coocurrència
% Aquests desplaçaments defineixen la distància i l’angle entre píxels:
% (0,d) = horitzontal, (-d,d) = diagonal dalt dreta, (-d,0) = vertical, (-d,-d) = diagonal dalt esquerra
offsets = [];
for d = 3:6
    offsets = [offsets;
               0 d;
              -d d;
              -d 0;
              -d -d];
end

% Calculem la matriu de coocurrència de nivells de gris (GLCM) per diversos desplaçaments
% graycomatrix crea una matriu que reflecteix la freqüència de combinacions de tons entre píxels veïns
% "Symmetric" indica que les coocurrències es compten de manera simètrica
cooc = graycomatrix(A,"Offset",offsets,"Symmetric",true);

% Extraiem estadístiques de textura a partir de les matrius de coocurrència
% Contrast: mesura la variació local d’intensitat
% Energy: mesura l’uniformitat (valors elevats per textures suaus)
% Homogeneity: valors alts indiquen distribució de tons similars entre píxels veïns
% Correlation: grau de correlació lineal entre píxels
stats = graycoprops(cooc, {'Contrast','Energy','Homogeneity','Correlation'});

% Calculem característiques estadístiques bàsiques:
% La mitjana d’intensitats (lluminositat mitjana)
mean_val = mean(A(:));
% La desviació estàndard (mesura la dispersió dels valors de gris)
std_val = std(double(A(:)));  % Convertim a double per fer el càlcul amb decimals

% Calculem un histograma simplificat:
% imhist crea un histograma amb 256 nivells de gris
hist_vals = imhist(A, 256);
% Reduïm el nombre de nivells de grisos a 8, interpolant les dades per tenir una representació compacta
hist8 = imresize(hist_vals, [8 1]);
% Normalitzem el vector de l’histograma perquè la suma sigui 1
hist8 = hist8 / sum(hist8);

% Concatenem totes les característiques en un únic vector:
% Incloem: característiques GLCM (contrast, energia, homogeneïtat, correlació),
% mitjana, desviació estàndard i histograma reduït
v = [stats.Contrast, stats.Energy, stats.Homogeneity, stats.Correlation, mean_val, std_val, hist8'];
