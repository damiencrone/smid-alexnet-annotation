% Executed using MATLAB 2016b


%% Setup

% Load the neural net
model = alexnet;
n_layers = numel(model.Layers);
sz = model.Layers(1).InputSize;

% Specify input directory and get image filenames
img_dir = '~/Desktop/img/';
dir_contents = dir(img_dir);
img_ind = ~[dir_contents.isdir];
dir_contents = dir_contents(img_ind);

% Specify number of labels to extract and calculate number of images
n_lab = 5;
n_img = numel(dir_contents);

% Create cell array of layer names and identify layers to be saved
layer_names = {model.Layers.Name};
fc_layers = find(~cellfun('isempty',strfind(layer_names, 'fc')));
prob_layer = find(strcmp(layer_names, 'prob'));
layers_to_save = [fc_layers];

% Initialise cell array of NaNs matrices for each layer
img_vectors = cell(1, numel(layers_to_save));
for i = 1:numel(layers_to_save)
    layer_num = layers_to_save(i);
    layer_size = model.Layers(layer_num).OutputSize;
    img_vectors{i} = nan(numel(dir_contents),layer_size);
end

img_labels = cell(numel(dir_contents), n_lab);


%% Describe images

for i = 1:n_img
    
    img_name = dir_contents(i).name;
    disp(img_name);
    
    try
        
        % Read the image to classify
        I = imread([img_dir img_name]);
        
        % If grayscale, convert to RGB
        if numel(size(I)) == 2
            I = cat(3, I, I, I);
        end
        
        % Adjust size of the image so smallest dimension matches alexnet
        if size(I, 1) > size(I, 2)
            I = imresize(I, [NaN sz(1)]);
        else
            I = imresize(I, [sz(1) NaN]);
        end
        
        % Crop larger dimension to match alexnet
        I = I(1:sz(1),1:sz(2),1:sz(3));
        
        for j = 1:numel(layers_to_save)
            layer_num = layers_to_save(j);
            img_vectors{j}(i,:) = activations(model, I, layer_num);
        end
        
        % Get probability layer
        Y = activations(model, I, prob_layer);
        [B,ind] = sort(Y,'descend');
        top_n_labels = model.Layers(25, 1).ClassNames(ind(1:n_lab));
        img_labels(i,:) = top_n_labels;
        
        label_str = strjoin(top_n_labels, ', ');
        disp(label_str);
        
    catch
        
    end
    
end


%% Save output

img_names = split({dir_contents.name}, '.');
img_names = {img_names{:,:,1}};

for i = 1:numel(layers_to_save)
    
    % Get relevant names and indices
    layer_num = layers_to_save(i);
    layer_name = layer_names{layer_num};
    prefix = ['alexnet_' layer_name '_'];
    
    % Construct table
    T = array2table(img_vectors{i}, 'RowNames', img_names);
    T.Properties.VariableNames = strrep(T.Properties.VariableNames, 'Var', prefix);
    
    % Save table to file
    table_fn = ['alexnet_' layer_name '.csv'];
    writetable(T, table_fn, 'WriteRowNames',true);
    
end

T = array2table(img_labels, 'RowNames', img_names);
table_fn = ['alexnet_top_' num2str(n_lab) '_labels.csv'];
writetable(T, table_fn, 'WriteRowNames',true);
