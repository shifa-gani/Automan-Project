function draw_object_triangle(imagePath, real_length_cm, real_height_cm, real_depth_cm)
    % Read and display image
    img = imread(imagePath);
    figure, imshow(img); title('Original Image');

    % Convert to grayscale and create a mask for non-white areas (object only)
    gray = rgb2gray(img);
    mask = ~(gray > 240); % Background is nearly white

    % Clean mask
    mask = bwareaopen(mask, 300); % Remove small regions
    mask = imfill(mask, 'holes');

    % Get properties of object
    stats = regionprops(mask, 'BoundingBox', 'PixelList');

    if isempty(stats)
        error('No object found!');
    end

    % Find largest object (in case there are tiny leftovers)
    [~, idx] = max(cellfun(@(x) size(x,1), {stats.PixelList}));
    pixels = stats(idx).PixelList;

    % Get topmost point (min y value, highest point of object)
    [~, topIdx] = min(pixels(:,2));
    topPoint = pixels(topIdx, :); % [x, y]

    % Get left and right bottom points from bounding box
    bbox = stats(idx).BoundingBox;
    leftBase = [bbox(1), bbox(2) + bbox(4)];
    rightBase = [bbox(1) + bbox(3), bbox(2) + bbox(4)];

    % Scale: convert image units (pixels) to cm
    base_pixel_length = norm(rightBase - leftBase);
    pixel_to_cm = real_length_cm / base_pixel_length;

    % Convert all points to real-world coordinates (in cm)
    A = (leftBase - leftBase) * pixel_to_cm;       % (0, 0)
    B = (rightBase - leftBase) * pixel_to_cm;      % (length, 0)
    C = (topPoint - leftBase) * pixel_to_cm;       % Top point in cm

    % Compute side lengths
    a = norm(B - C); % side opposite to A
    b = norm(C - A); % side opposite to B
    c = norm(B - A); % base (should be ~real_length_cm)

    % Compute angles using cosine rule
    angleA = acosd((b^2 + c^2 - a^2) / (2*b*c));
    angleB = acosd((a^2 + c^2 - b^2) / (2*a*c));
    angleC = 180 - angleA - angleB;

    % Plot triangle on image
    figure, imshow(img); title('Triangle Overlaid'); hold on;
    plot([leftBase(1), rightBase(1)], [leftBase(2), rightBase(2)], 'y', 'LineWidth', 2); % base
    plot([leftBase(1), topPoint(1)], [leftBase(2), topPoint(2)], 'y', 'LineWidth', 2); % left side
    plot([rightBase(1), topPoint(1)], [rightBase(2), topPoint(2)], 'y', 'LineWidth', 2); % right side
    plot(topPoint(1), topPoint(2), 'ro', 'MarkerFaceColor', 'r'); % mark top

    % Display angles
    fprintf('\n--- Triangle Angle Measurements ---\n');
    fprintf('Angle at Left Base (A): %.2f degrees\n', angleA);
    fprintf('Angle at Right Base (B): %.2f degrees\n', angleB);
    fprintf('Angle at Top Point (C): %.2f degrees\n', angleC);

end

draw_object_triangle("C:\Users\HP\Desktop\Automan_proj\trial.png",18, 13.34, 15.62);

% Triangle parameters
base = 18;
height = 13.34;
depth = 15.62;  % Prism thickness

% Assume base lies on X-axis from (0,0) to (18,0)
A = [0, 0];
B = [base, 0];

% Use law of sines to find the length of side AC (opposite angle B)
AC = base * sin(angleB) / sin(angleC);

% Coordinates of point C (opposite base AB)
Cx = AC * cos(angleA);
Cy = AC * sin(angleA);
C = [Cx, Cy];

% Triangle vertices
triangle = [A; B; C];

L0 = base;
H0 = height;
W = depth;

originalSurfaceArea = 2 * (L0*W + W*H0 + H0*L0);

% --- Optimization over rotation ---
minSurfaceArea = originalSurfaceArea;
bestAngle = 0;
bestDims = [L0, H0];

for angle = 0:0.1:90  % Start from 0.1 to skip original case
    theta = deg2rad(angle);
    centroid = mean(triangle);  % rotate about centroid

    R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    rotated = (triangle - centroid) * R' + centroid;

    minX = min(rotated(:,1));
    maxX = max(rotated(:,1));
    minY = min(rotated(:,2));
    maxY = max(rotated(:,2));

    L = maxX - minX;
    H = maxY - minY;

    surfaceArea = 2 * (L*W + W*H + H*L);

    if surfaceArea < minSurfaceArea
        minSurfaceArea = surfaceArea;
        bestAngle = angle;
        bestDims = [L, H];
    end
end

% --- Reporting ---
fprintf('Original box dimensions (L x H x W): %.2f x %.2f x %.2f cm\n', L0, H0, depth);
fprintf('Original surface area: %.2f cm²\n\n', originalSurfaceArea);

fprintf('Best rotation angle: %.2f°\n', bestAngle);
fprintf('Optimized box dimensions (L x H x W): %.2f x %.2f x %.2f cm\n', ...
        bestDims(1), bestDims(2), depth);
fprintf('Minimum surface area: %.2f cm²\n', minSurfaceArea);
fprintf('Surface area reduced by: %.2f%%\n', ...
        100 * (originalSurfaceArea - minSurfaceArea) / originalSurfaceArea);


