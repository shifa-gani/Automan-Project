function compare_triangle_fit(imagePath, real_length_top, real_height_side)

    % Read image
    img = imread(imagePath);
    
    % Split into two views (left and right halves)
    img_top = img(:, 1:floor(size(img,2)/2), :);
    img_side = img(:, floor(size(img,2)/2)+1:end, :);
    
    % Process top view
    fprintf('Processing Top View...\n');
    [fit_error_top, area_top] = draw_object_triangle(img_top, real_length_top);
    
    % Process side view
    fprintf('Processing Side View...\n');
    [fit_error_side, area_side] = draw_object_triangle(img_side, real_height_side);
    
    % Compare
    fprintf('\nComparison Result:\n');
    fprintf('Top View Triangle Fit Error: %.2f\n', fit_error_top);
    fprintf('Side View Triangle Fit Error: %.2f\n', fit_error_side);

    if fit_error_top < fit_error_side
        fprintf('=> Top View has better triangle fitting.\n');
    else
        fprintf('=> Side View has better triangle fitting.\n');
    end
end

function [fit_error, area_cm2] = draw_object_triangle(img, real_length_cm)
    % Display image
    figure, imshow(img); title('Original View');

    % Convert to grayscale and mask non-white
    gray = rgb2gray(img);
    mask = ~(gray > 240);

    % Clean mask
    mask = bwareaopen(mask, 300);
    mask = imfill(mask, 'holes');

    % Get object properties
    stats = regionprops(mask, 'BoundingBox', 'PixelList');

    if isempty(stats)
        error('No object found!');
    end

    [~, idx] = max(cellfun(@(x) size(x,1), {stats.PixelList}));
    pixels = stats(idx).PixelList;

    % Find topmost point
    [~, topIdx] = min(pixels(:,2));
    topPoint = pixels(topIdx, :);

    % Bounding box for base
    bbox = stats(idx).BoundingBox;
    leftBase = [bbox(1), bbox(2) + bbox(4)];
    rightBase = [bbox(1) + bbox(3), bbox(2) + bbox(4)];

    % Scale pixels to cm
    base_pixel_length = norm(rightBase - leftBase);
    pixel_to_cm = real_length_cm / base_pixel_length;

    A = (leftBase - leftBase) * pixel_to_cm;      % (0,0)
    B = (rightBase - leftBase) * pixel_to_cm;     % (length,0)
    C = (topPoint - leftBase) * pixel_to_cm;       % Top point in cm

    % Calculate sides
    a = norm(B - C);
    b = norm(C - A);
    c = norm(B - A);

    % Calculate triangle area
    s = (a+b+c)/2;
    area_cm2 = sqrt(s*(s-a)*(s-b)*(s-c));

    % Find pixel mask boundary
    boundary = bwboundaries(mask);
    boundary_points = boundary{1};

    % Calculate how close boundary points are to triangle edges
    dists = [];
    for i = 1:size(boundary_points,1)
        p = boundary_points(i,:);
        p_cm = (p - [leftBase(2), leftBase(1)]) * pixel_to_cm; % [row,col] to [y,x] flip
        p_cm = [p_cm(2), -p_cm(1)]; % Adjust axes
        
        % Distances to triangle edges
        d1 = point_to_line_distance(p_cm, A, C);
        d2 = point_to_line_distance(p_cm, C, B);
        d3 = point_to_line_distance(p_cm, A, B);
        
        dists(end+1) = min([d1,d2,d3]);
    end
    
    % Mean distance error
    fit_error = mean(dists);

    % Plot triangle
    figure, imshow(img); hold on;
    plot([leftBase(1), rightBase(1)], [leftBase(2), rightBase(2)], 'g-', 'LineWidth', 2);
    plot([leftBase(1), topPoint(1)], [leftBase(2), topPoint(2)], 'g-', 'LineWidth', 2);
    plot([rightBase(1), topPoint(1)], [rightBase(2), topPoint(2)], 'g-', 'LineWidth', 2);
    title('Triangle Fitted on View');
end

function d = point_to_line_distance(P, A, B)
    % Distance from point P to line segment AB
    AB = B - A;
    AP = P - A;
    t = max(0, min(1, dot(AP,AB)/dot(AB,AB)));
    proj = A + t*AB;
    d = norm(P - proj);
end
compare_triangle_fit("C:\Users\HP\Downloads\WhatsApp Image 2025-04-16 at 14.11.24.jpeg", 10, 8);
