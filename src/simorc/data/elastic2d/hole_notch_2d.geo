//==============================================================================
// Gmsh 2D parametric plate mesh with circular edge notch and hole
// author: Lloyd Fletcher (scepticalrabbit)
//==============================================================================

// Always set to OpenCASCADE - circles and boolean opts are much easier!
SetFactory("OpenCASCADE");

// Allows gmsh to print to terminal in vscode - easier debugging
General.Terminal = 0;

// View options
Geometry.PointLabels = 0;
Geometry.CurveLabels = 0;
Geometry.SurfaceLabels = 0;
Geometry.VolumeLabels = 0;

//------------------------------------------------------------------------------
// Variables

//_* MOOSEHERDER VARIABLES - START
file_name = "hole_notch_2d.msh";

// Geometric variables
plate_width = 25.0;
plate_height = 35.0; // Must be greater than plate width

plate_diff = plate_height-plate_width;

// Notch variables (left edge)
notch_rad = 5.0;
notch_loc_x = -plate_width/2;
notch_loc_y = plate_height/2;

// Hole variables (right)
hole_rad = 5.0;
hole_loc_x = 2.5;
hole_loc_y = plate_height/2;

// Mesh variables
mesh_ref = 4;
mesh_size = 4.0/mesh_ref;

tol = mesh_size/4; // Used for bounding box selection tolerance

second_ord_incomp = 1;
notch_ref_factor = 2.0;
hole_ref_factor = 2.0;
//** MOOSEHERDER VARIABLES - END

//------------------------------------------------------------------------------
// Geometry Definition

// Split plate into eight pieces following the original convention.
s1 = news;
Rectangle(s1) = {-plate_width/2,0.0,0.0,
                plate_width/2,plate_diff/2};

s2 = news;
Rectangle(s2) = {0.0,0.0,0.0,
                plate_width/2,plate_diff/2};

s3 = news;
Rectangle(s3) = {-plate_width/2,plate_diff/2,0.0,
                plate_width/2,plate_width/2};

s4 = news;
Rectangle(s4) = {0.0,plate_diff/2,0.0,
                plate_width/2,plate_width/2};

s5 = news;
Rectangle(s5) = {-plate_width/2,plate_width/2+plate_diff/2,0.0,
                plate_width/2,plate_width/2};

s6 = news;
Rectangle(s6) = {0.0,plate_width/2+plate_diff/2,0.0,
                plate_width/2,plate_width/2};

s7 = news;
Rectangle(s7) = {-plate_width/2,plate_height-plate_diff/2,0.0,
                plate_width/2,plate_diff/2};

s8 = news;
Rectangle(s8) = {0.0,plate_height-plate_diff/2,0.0,
                plate_width/2,plate_diff/2};

// Merge coincident edges of the overlapping rectangles
BooleanFragments{ Surface{s1}; Delete; }
                { Surface{s2,s3,s4,s5,s6,s7,s8}; Delete; }

// Create notch cutting surface
c1 = newc;
Circle(c1) = {notch_loc_x,notch_loc_y,0.0,notch_rad};

cl1 = newcl;
Curve Loop(cl1) = {c1};

sn1 = news;
Plane Surface(sn1) = {cl1};

// Create hole cutting surface
c2 = newc;
Circle(c2) = {hole_loc_x,hole_loc_y,0.0,hole_rad};

cl2 = newcl;
Curve Loop(cl2) = {c2};

sn2 = news;
Plane Surface(sn2) = {cl2};

// Cut the circular edge notch and hole out of the full plate
BooleanDifference{ Surface{:}; Delete; }
                 { Surface{sn1,sn2}; Delete; }

//------------------------------------------------------------------------------
// Mesh sizing

// Global characteristic length
Mesh.CharacteristicLengthMin = mesh_size / 2.0;
Mesh.CharacteristicLengthMax = mesh_size;

// Mesh Refinement Fields around notch and hole
Field[1] = Box;
Field[1].VIn = mesh_size / notch_ref_factor;
Field[1].VOut = mesh_size;
Field[1].XMin = notch_loc_x - notch_rad - tol;
Field[1].XMax = notch_loc_x + notch_rad + tol;
Field[1].YMin = notch_loc_y - notch_rad - tol;
Field[1].YMax = notch_loc_y + notch_rad + tol;
Field[1].ZMin = -1.0;
Field[1].ZMax = 1.0;
Field[1].Thickness = mesh_size * 2;

Field[2] = Box;
Field[2].VIn = mesh_size / hole_ref_factor;
Field[2].VOut = mesh_size;
Field[2].XMin = hole_loc_x - hole_rad - tol;
Field[2].XMax = hole_loc_x + hole_rad + tol;
Field[2].YMin = hole_loc_y - hole_rad - tol;
Field[2].YMax = hole_loc_y + hole_rad + tol;
Field[2].ZMin = -1.0;
Field[2].ZMax = 1.0;
Field[2].Thickness = mesh_size * 2;

Field[3] = Min;
Field[3].FieldsList = {1, 2};
Background Field = 3;

//------------------------------------------------------------------------------
// Physical lines and surfaces for export/BCs

Physical Surface("plate") = {Surface{:}};

pc1() = Curve In BoundingBox{
    -plate_width/2-tol,0.0-tol,0.0-tol,
    plate_width/2+tol,0.0+tol,0.0+tol};
Physical Curve("bc-bot") = {pc1()};

pc2() = Curve In BoundingBox{
    -plate_width/2-tol,plate_height-tol,0.0-tol,
    plate_width/2+tol,plate_height+tol,0.0+tol};
Physical Curve("bc-top") = {pc2()};

// Mid-points on top and bottom boundaries for pinning X displacement
p_bot() = Point In BoundingBox{-tol, -tol, -tol, tol, tol, tol};
Physical Point("bc-bot-mid") = {p_bot(0)};

p_top() = Point In BoundingBox{-tol, plate_height-tol, -tol, tol, plate_height+tol, tol};
Physical Point("bc-top-mid") = {p_top(0)};

//------------------------------------------------------------------------------
// Global meshing

Mesh.ElementOrder = 2;
Mesh.SecondOrderIncomplete = second_ord_incomp;

Mesh 2;

//------------------------------------------------------------------------------
// Save and exit

Save Str(file_name);
Exit;
