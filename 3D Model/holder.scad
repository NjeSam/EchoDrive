w = 140;
h = 90; 
thickness = 10;
fillet = 5; 
hole_dia = 5; 
hole_rad = hole_dia/2;

$fn = 100;

animation_factor = ( 1 + cos(360 * $t)) / 2;
amplitude = 23.5;

coil_diameter = 2;
wire_diameter = 2;
num_coils = 12; 
min_height = 20.5;
max_height = 44; 
spring_height = min_height + (max_height - min_height) * animation_factor; 

coil_radius = coil_diameter / 2;
wire_radius = wire_diameter / 2;
total_twist = 360 * num_coils; 

module filleted_block() {
    minkowski() {
        cube([w, h, thickness], center = false);
        sphere(r = fillet);
    }
}

module bottom() {
    translate([5, 5, -10]) union() {
        difference() {
            filleted_block();

            translate([-fillet, -fillet, thickness])
                cube([w + 2*fillet, h + 2*fillet, fillet], center = false);

            offset = fillet*2 + hole_rad;

            for (xpos = [offset, w - offset], ypos = [offset, h - offset]) {
                translate([xpos, ypos, -11])
                    cylinder(h = thickness + 15, r = hole_rad);
            }

            for (xpos = [offset, w - offset], ypos = [offset, h - offset]) {
                translate([xpos, ypos, 6])
                    cylinder(h = thickness, r = hole_dia);
            }

            for (xpos = [offset, w - offset], ypos = [offset, h - offset]) {
                translate([xpos, ypos, -11])
                    cylinder(h = thickness, r = hole_dia);
            }

            translate([25, 20, 2]) cube([85, 50, 15]);

            translate([85, 45, -2]) cylinder(h = thickness, r = hole_dia);
        }

        translate([85, 45, -2]) cylinder(h = thickness, r = hole_rad);

        difference() {
            translate([-5, 37.5, 4]) cube([30, 15, 11]); 
            translate([6, 42.5, 5]) cube([25, 5, 11]);   
        }
    }
}

module top_cap() {
    translate([5, 5, -5]) union() {
        difference() {
        filleted_block();

        translate([-fillet, -fillet, thickness-15])
            cube([w + 2*fillet, h + 2*fillet, fillet*2]);
            
        translate([-5, 20, 0]) cube([100, 50, 10]);
        translate([55, -5, 0]) cube([60, 140, 10]); 
        }

        difference() {
            offset = fillet*2 + hole_rad;
            for (xpos = [offset, w - offset], ypos = [offset, h - offset]) {
                translate([xpos, ypos, 0]) cylinder(h = thickness, r = hole_dia);
            }
            for (xpos = [offset, w - offset], ypos = [offset, h - offset]) {
                translate([xpos, ypos, -1]) cylinder(h = thickness, r = hole_rad);
            }
        }
    }
}

module swivel() {
    rotate([0, 0, 135 - 65 * (1 - animation_factor)]) union() {
        linear_extrude(height = 4) difference() {
            circle(d = 50);
            translate([0, 0, 4]) circle(d = hole_dia+2);
        }

        translate([0, 0, -8]) linear_extrude(height = thickness-2) difference() {
            circle(d = hole_dia*2); 
            circle(d = hole_dia);   
        }

        translate([0, 0, -4]) linear_extrude(height = thickness-2) difference() {
            circle(d = hole_dia*2+2);
            circle(d = hole_dia);
        }

        translate([-21, 0, -4]) linear_extrude(height = thickness-2) circle(d = hole_dia);
        translate([0, -22.5, 0]) linear_extrude(height = thickness-2) circle(d = hole_dia);
        translate([0, 22.5, 0]) linear_extrude(height = thickness-2) circle(d = hole_dia);
    }
}

module handle() {
    rotate([90, 0, 90]) difference() {
        linear_extrude(height = 60) offset(r = fillet) square([80, 13]);

        translate([-21, 1.5, -10]) linear_extrude(height = 100)
            offset(r = fillet/2) square([100, 10]);
        translate([-15, 2, -10]) linear_extrude(height = 100)
            offset(r = fillet/2) square([70, 18]);
        translate([-55, -8, 40]) linear_extrude(height = 100) square([70, 10]);
        
        translate([-60, -10, -8]) cube([60, 50, 50]);
        translate([-10, -10, 41]) rotate([0, 90, 0]) cube([11, 10, 70]);
        translate([19.5, -10, 55]) rotate([0, 90, 0]) cube([12, 10, 6.5]);
        translate([-10, -10, 19]) rotate([0, 90, 0]) cube([30, 10, 54]);
    }
}

module lower_handle(){
    union(){
        rotate([90,0,180])
        difference(){
            linear_extrude(height = 50)
            offset(r = fillet)
            square([100, 15]);

            translate([-2,2.5,-10])
            linear_extrude(height = 100)
            offset(r = fillet/2)
                square([100, 10]);
            
            translate([2,5,-10])
            linear_extrude(height = 100)
            offset(r = fillet/2)
                square([70, 18]);
            
            translate([-55,-10,-8])
            cube([60,50,60]);
            
            translate([10,-12,32.5])
            rotate([0,90,0])
            cube([15,15,55]);
            
            translate([70,6 ,32.5])
            rotate([0,90,0])
            cube([15,30,38]);
        }

        translate([-10,0,-13]) cube([5,50,8]);

        difference(){
            translate([-5,0,-13]) cube([55,50,4]);
            translate([15,18.5,-15]) cube([40,13,8]);
            translate([37.5,1.2,-15]) cube([6,11.5,8]);
        }
    }
}

module hook(){
    union(){
        translate([80,35,-75]) cube([5,30,60]);

        difference(){
            translate([73.5,35,-75]) cube([5,30,60]);
            translate([73.5,34,-75]) rotate([0,45,0]) cube([5,35,8]);
        }
        
        translate([85,35,-75]) rotate([0,90,90])
        linear_extrude(30, center = false) {
            right_triangle(6, 8);
        }
    }
}

module right_triangle(a, b) {
    polygon(points=[
        [0, 0],
        [a, 0],
        [0, b]
    ], paths=[[0,1,2]]);
}

module spring(){
    linear_extrude(height = spring_height, twist = total_twist, slices = 200) {
        translate([coil_radius, 0]) 
        circle(r = wire_radius, $fn=50);
    }
}

color("Gold") {
    bottom(); 
    top_cap();
    hook();
}

color("red") {
    translate([60, 43.5 - amplitude * (1 - animation_factor), 5.001]) handle();
    translate([120, 56.5 + amplitude * (1 - animation_factor), 5.001]) rotate([0, 0, 180]) handle();
}

translate([90, 50, -4]) color("cyan") swivel();

translate([65 - amplitude * (1 - animation_factor), 25, 5.001]) color("indigo") lower_handle();
translate([11,50,3]) rotate([0,90,0]) color("white") spring();