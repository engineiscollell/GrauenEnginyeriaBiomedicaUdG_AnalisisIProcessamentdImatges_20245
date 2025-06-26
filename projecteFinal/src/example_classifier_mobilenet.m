function example_classifier_mobilenet
% Afegim la carpeta VOCcode al path
addpath([cd '/VOCcode']);
% Carreguem la xarxa AlexNet preentrenada
net = mobilenetv2;
net.Layers
net.Layers(end-2)  % Capa Logits
featureLayer = 'global_average_pooling2d_1'; %extreurem característiques de les diferents capes

% Inicialitzem les opcions VOC
VOCinit;
for i=1:VOCopts.nclasses
    cls = VOCopts.classes{i};
    classifier = train(VOCopts, cls, net, featureLayer); % entrenem amb SVM
    test(VOCopts, cls, classifier, net, featureLayer);   % test amb SVM
    [fp, tp, auc] = VOCroc(VOCopts, 'comp1', cls, true); % corba ROC
    allFP{i} = fp;%modifiquem per collage
    allTP{i} = tp;
    allAUC(i) = auc;
    if i < VOCopts.nclasses
        fprintf('press any key to continue with next class...\n');
        pause;
    end
end



fprintf('Mostrem collage Corbes ROC + Gràfic Lineal per visualitzar més còmodament les dades:');


% Collage de corbes ROC
figure;
for i = 1:VOCopts.nclasses
    subplot(2,5,i);
    plot(allFP{i}, allTP{i}, 'b-', 'LineWidth', 2);
    axis([0 1 0 1]);
    xlabel('FPR');
    ylabel('TPR');
    title(sprintf('%s (AUC=%.2f)', VOCopts.classes{i}, allAUC(i)));
    grid on;
end
sgtitle('Corbes ROC per classe (MobileNet-v2, global_average_pooling2d_1, K-NN, NumNeighbors=3)');

% Gràfic lineal
figure;
plot(allAUC, '-o');
xticks(1:VOCopts.nclasses);
xticklabels(VOCopts.classes);
xtickangle(45);
ylabel('AUC');
title('AUC per classe (MobileNet-v2, global_average_pooling2d_1, K-NN, NumNeighbors=3)');
grid on;


% ENTRENAMENT DEL CLASSIFICADOR
function classifier = train(VOCopts,cls, net, featureLayer)
    [ids,classifier.gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,'train'),'%s %d');
    tic;
    % Pre-allocate feature matrix with the correct dimensions
    classifier.FD = [];
    for i=1:length(ids)
        if toc>1
            fprintf('%s: train: %d/%d\n',cls,i,length(ids));
            drawnow;
            tic;
        end
        try
            load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        catch
            I=imread(sprintf(VOCopts.imgpath,ids{i}));
            fd=extractfd(I, net, featureLayer);
            save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        end
        
        % Make sure fd is a column vector
        fd = fd(:);
        
        % On first iteration, initialize FD matrix with correct dimensions
        if i == 1
            % Use single precision for feature data to match what kNN expects
            classifier.FD = zeros(length(fd), length(ids), 'single');
        end
        
        % Ensure consistent data type
        fd = single(fd);
        classifier.FD(:,i) = fd;
    end
    
    classifier.model = fitcknn(classifier.FD', classifier.gt', 'NumNeighbors', 3); % Canviar per SVM o K-NN
  

% TEST DEL CLASSIFICADOR
function test(VOCopts,cls,classifier, net, featureLayer)
    [ids,gt]=textread(sprintf(VOCopts.clsimgsetpath,cls,VOCopts.testset),'%s %d');
    fid=fopen(sprintf(VOCopts.clsrespath,'comp1',cls),'w');
    tic;
    for i=1:length(ids)
        if toc>1
            fprintf('%s: test: %d/%d\n',cls,i,length(ids));
            drawnow;
            tic;
        end
        try
            load(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        catch
            I=imread(sprintf(VOCopts.imgpath,ids{i}));
            fd=extractfd(I, net, featureLayer);
            save(sprintf(VOCopts.exfdpath,ids{i}),'fd');
        end
        c=classify(classifier,fd); % classificació amb kNN
        fprintf(fid,'%s %f\n',ids{i},c);
    end
    fclose(fid);

% EXTRACCIÓ DE CARACTERÍSTIQUES AMB ALEXNET
function fd = extractfd(I, net, featureLayer)
    inputSize = net.Layers(1).InputSize;
    I = imresize(I, [inputSize(1), inputSize(2)]);
    % Convertir a RGB si és en escala de grisos
    if size(I,3) == 1
        I = repmat(I, [1 1 3]);
    end
    features = activations(net, I, featureLayer, 'OutputAs', 'columns');  %%% MODIFICAT PER RESNET-101
    fd = single(features);  % ja és vector columna %%% MODIFICAT PER RESNET-101
    
    % Ensure consistent feature dimensionality
    expectedLength = 1280;  % MobileNet-v2
    if numel(fd) ~= expectedLength
        warning('Warning a extractfd: el vector de característiques no té mida %d', numel(fd), expectedLength);
        fd = zeros(expectedLength, 1, 'single');
    end

% CLASSIFICADOR kNN
function c = classify(classifier, fd)
    % Ensure fd is a row vector with the same number of columns as expected
    fd = fd(:)';  % Convert to row vector 
    
    % Check dimensions
    if size(fd, 2) ~= size(classifier.FD, 1)
        error('Error: Feature dimension mismatch. Expected %d features but got %d.', ...
            size(classifier.FD, 1), size(fd, 2));
    end
    
    % Explicitly convert to the same data type as training data to avoid warnings
    trainType = class(classifier.FD);
    fd = cast(fd, trainType);
    
    [~, score] = predict(classifier.model, fd); % predicció amb SVM
    c = score(2); % puntuació per la classe positiva
