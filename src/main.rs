// Quantum wavepacket moving in 2D
// More comments forthcoming
use macroquad::prelude::*;
use macroquad::texture::Image;
use num::complex::Complex;
use std::fs;
use std::io::Write;
use std::vec::Vec;

//use std::f64::consts::PI;

const NO: usize = 1000;

const H2M0: f64 = 7.62e-2;

const WIDTH: i32 = 1000;
const HEIGHT: i32 = 1000;

fn window_conf() -> Conf {
    Conf {
        window_title: "Wavepacket trapped".to_owned(),
        window_width: WIDTH,
        window_height: HEIGHT,
        ..Default::default()
    }
}

fn _taxicab_dist (x: f64, y: f64) -> f64 {
    ((x+y).abs() + (x-y).abs())/std::f64::consts::SQRT_2
}

fn eucl_dist (x: f64, y: f64) -> f64 {
    (x.powf(2.0) + y.powf(2.0)).sqrt()
}

struct Params {
    grid_lx: f64,
    grid_ly: f64,
    grid_nx: usize,
    wp_eff_mass: f64,
    wp_x0: f64,
    wp_y0: f64,
    wp_a: f64,
    pot_u0: f64,
    pot_x0: f64,
    pot_y0: f64,
    pot_r1: f64,
    pot_r2: f64,
    calc_time: f64,
    nt: usize,
}

impl Params {
    fn create() -> Params {
        Params {
            grid_lx: 35.0,
            grid_ly: 35.0,
            grid_nx: 1000,
            wp_eff_mass: 0.067,
            wp_x0: 0.0,
            wp_y0: 0.0,
            wp_a: 20.0,
            pot_u0: 0.2,
            pot_x0: 0.0,
            pot_y0: 0.0,
            pot_r1: 26.0,
            pot_r2: 28.0,
            calc_time: 100.0,
            nt: 1000,
        }
    }
}

// Граничные точки по x будут иметь индексы 0 и 2*nx+1
// Граничные точки по y будут иметь индексы 0 и 2*ny+3 (граничные у-я)
// В граничных точках волновая функция равна нулю
struct Grid {
    lx: f64,
    ly: f64,
    nx: usize,
    ny: usize,
    width: usize,
    height: usize,
    dx: f64,
    dy: f64,
    rdx2: f64,
    rdy2: f64,
}

impl Grid {
    fn create(params: &Params) -> Grid {
        let ny = params.grid_nx;
        let dx = params.grid_lx / (1.0 + params.grid_nx as f64);
        let dy = params.grid_ly / (1.0 + ny as f64);

        Grid {
            lx: params.grid_lx,
            ly: params.grid_ly,
            nx: params.grid_nx,
            ny,
            width: 2 * params.grid_nx + 3,
            height: 2 * ny + 3,
            dx,
            dy,
            rdx2: 1.0 / (dx * dx),
            rdy2: 1.0 / (dy * dy),
        }
    }
}

#[derive(Clone)]
struct Wavepacket {
    x0: f64,
    y0: f64,
    a: f64,
    psi_grid: Vec<Complex<f64>>,
    norm: f64,
    psi_max: f64,
    psi_min: f64,
}

impl Wavepacket {
    fn create(grid: &Grid, params: &Params) -> Wavepacket {
        let mut w = Wavepacket {
            x0: params.wp_x0,
            y0: params.wp_y0,
            a: params.wp_a,
            psi_grid: Vec::new(),
            norm: 0.0,
            psi_max: 0.0,
            psi_min: 0.0,
        };
        let result = w.initial_state_grid(grid);
        w.psi_grid = result.0;
        w.norm = w.get_norm(grid);
        w.psi_min = result.1;
        w.psi_max = result.2;
        w
    }

    fn get_norm(&self, grid: &Grid) -> f64 {
        let mut norm = 0.0;
        for jx in 0..grid.width {
            for jy in 0..grid.height {
                norm += self.psi_grid[jx * grid.height + jy].norm_sqr()
            }
        }
        norm
    }

    fn initial_state(&self, x: f64, y: f64) -> Complex<f64> {
        let wy = (-((y - self.y0) / self.a).powf(2.0)).exp();
        let wx = (-((x - self.x0) / self.a).powf(2.0)).exp();
        let mut psi = num::complex::Complex::new(1.0, 0.0);
        psi = psi.scale(wx * wy);
        psi
    }

    fn initial_state_grid(&self, grid: &Grid) -> (Vec<Complex<f64>>, f64, f64) {
        let mut psi_grid: Vec<Complex<f64>> = Vec::new();
        let mut psi_min = self.psi_min;
        let mut psi_max = self.psi_max;
        for jx in 0..grid.width {
            for jy in 0..grid.height {
                let x = jx as f64 * grid.dx - grid.lx - grid.dx;
                let y = jy as f64 * grid.dy - grid.ly - grid.dy;
                let val = self.initial_state(x, y);

                psi_grid.push(val);

                let norm = val.norm_sqr();

                if norm < psi_min {
                    psi_min = norm
                }

                if norm > psi_max {
                    psi_max = norm
                }
            }
        }
        (psi_grid, psi_min, psi_max)
    }

    fn plot_2d(&self, grid: &Grid) -> Image {
        let image_width = grid.width as u16;
        let image_height = grid.height as u16;

        let mut image = Image::gen_image_color(
            image_width,
            image_height,
            macroquad::color::Color::from_rgba(255, 255, 255, 255),
        );

        for jx in 0..(image_width) {
            for jy in 0..(image_height) {
                let jgrid = ((jx as usize)) * grid.height + ((jy as usize));
                let z = self.psi_grid[jgrid];
                let col = palette(z.norm_sqr(), self.psi_min, self.psi_max);
                image.set_pixel(jx as u32, jy as u32, col);
            }
        }

        image
    }
}

struct Potential {
    u0_scaled: f64,
    radius_1: f64,
    radius_2: f64,
    x0: f64,
    y0: f64,
    u_grid: Vec<f64>,
    u_min: f64,
    u_max: f64,
}

impl Potential {
    fn create(grid: &Grid, params: &Params) -> Potential {
        // В версии программы на R вместо этой амплитуды использовалась исходная в эВ, это была ошибка
        //let u0_scaled = params.pot_u0;
        let u0_scaled = 2.0 * params.wp_eff_mass / H2M0 * params.pot_u0;
        let mut p = Potential {
            u0_scaled,
            radius_1: params.pot_r1,
            radius_2: params.pot_r2,
            x0: params.pot_x0,
            y0: params.pot_y0,
            u_grid: Vec::new(),
            u_min: 0.0,
            u_max: u0_scaled,
        };
        let values = p.get_values_grid(grid);
        p.u_grid = values.0;
        p.u_min = values.1;
        p.u_max = values.2;
        p
    }

    fn get_value(&self, x: f64, y: f64) -> f64 {
        let s = eucl_dist(x - self.x0, y - self.y0);
        let mut value = 0.0;
        if s > self.radius_1 && s < self.radius_2 {
            value = self.u0_scaled
        }
        value
    }

    fn get_values_grid(&self, grid: &Grid) -> (Vec<f64>, f64, f64) {
        let mut u_grid: Vec<f64> = Vec::new();
        let mut u_min = self.u_min;
        let mut u_max = self.u_max;
        for jx in 0..grid.width {
            for jy in 0..grid.height {
                let x = jx as f64 * grid.dx - grid.lx - grid.dx;
                let y = jy as f64 * grid.dy - grid.ly - grid.dy;
                let val = self.get_value(x, y);
                u_grid.push(val);

                if val < u_min {
                    u_min = val
                };
                if val > u_max {
                    u_max = val
                };
            }
        }
        (u_grid, u_min, u_max)
    }

    fn plot_2d(&self, grid: &Grid) {
        let fl_name = format!("{}_{}.png", "potential", NO);

        let image_width = grid.width as u16;
        let image_height = grid.height as u16;

        let mut image = Image::gen_image_color(
            image_width,
            image_height,
            macroquad::color::Color::from_rgba(255, 255, 255, 0),
        );

        for jx in 0..(image_width) {
            for jy in 0..(image_height) {
                let jgrid = ((jx as usize)) * grid.height + ((jy as usize));
                let u = self.u_grid[jgrid];
                if u > 1e-12 {
                    image.set_pixel(jx as u32, jy as u32, LIGHTGRAY);
                }
                
            }
        }
        image.export_png(&fl_name)

    }
}

struct GlobalState {
    nt: usize,
    rdt: f64,
    grid: Grid,
    pot: Potential,
    wp1: Wavepacket,
    wp2: Wavepacket,
}

impl GlobalState {
    fn create(params: &Params) -> GlobalState {
        let grid = Grid::create(params);
        let pot = Potential::create(&grid, params);
        let wp1 = Wavepacket::create(&grid, params);
        let wp2 = wp1.clone();
        let calc_time = params.calc_time;
        let dt = calc_time / 2.0 / (1.0 + params.nt as f64);
        GlobalState {
            nt: params.nt,
            rdt: 1.0 / dt,
            grid,
            pot,
            wp1,
            wp2,
        }
    }

    fn update(&mut self) {
        // Идем слева
        for jx in 0..self.grid.nx {
            // Идем вниз
            for jy in 0..(2 * self.grid.ny) {
                let jgrid = (2 * jx + 1) * self.grid.height + jy + 1;
                let pxy = num::complex::Complex::new(
                    0.5 * self.pot.u_grid[jgrid] + self.grid.rdx2 + self.grid.rdy2,
                    self.rdt,
                );

                self.wp2.psi_grid[jgrid] = -(pxy * self.wp1.psi_grid[jgrid]
                    - self.grid.rdx2
                        * (self.wp2.psi_grid[jgrid - self.grid.height]
                            + self.wp1.psi_grid[jgrid + self.grid.height])
                    - self.grid.rdy2
                        * (self.wp2.psi_grid[jgrid - 1] + self.wp1.psi_grid[jgrid + 1]))
                    / pxy.conj();
            }
            // Идем вверх
            for jy in (0..(2 * self.grid.ny)).rev() {
                let jgrid = (2 * jx + 2) * self.grid.height + jy + 1;
                let pxy = num::complex::Complex::new(
                    0.5 * self.pot.u_grid[jgrid] + self.grid.rdx2 + self.grid.rdy2,
                    self.rdt,
                );

                self.wp2.psi_grid[jgrid] = -(pxy * self.wp1.psi_grid[jgrid]
                    - self.grid.rdx2
                        * (self.wp2.psi_grid[jgrid - self.grid.height]
                            + self.wp1.psi_grid[jgrid + self.grid.height])
                    - self.grid.rdy2
                        * (self.wp1.psi_grid[jgrid - 1] + self.wp2.psi_grid[jgrid + 1]))
                    / pxy.conj();
            }
        }

        self.wp1 = self.wp2.clone();

        // Идем справа
        for jx in (0..self.grid.nx).rev() {
            // Идем вниз
            for jy in 0..(2 * self.grid.ny) {
                let jgrid = (2 * jx + 2) * self.grid.height + jy + 1;
                let pxy = num::complex::Complex::new(
                    0.5 * self.pot.u_grid[jgrid] + self.grid.rdx2 + self.grid.rdy2,
                    self.rdt,
                );

                self.wp2.psi_grid[jgrid] = -(pxy * self.wp1.psi_grid[jgrid]
                    - self.grid.rdx2
                        * (self.wp1.psi_grid[jgrid - self.grid.height]
                            + self.wp2.psi_grid[jgrid + self.grid.height])
                    - self.grid.rdy2
                        * (self.wp2.psi_grid[jgrid - 1] + self.wp1.psi_grid[jgrid + 1]))
                    / pxy.conj();
            }
            // Идем вверх
            for jy in (0..(2 * self.grid.ny)).rev() {
                let jgrid = (2 * jx + 1) * self.grid.height + jy + 1;
                let pxy = num::complex::Complex::new(
                    0.5 * self.pot.u_grid[jgrid] + self.grid.rdx2 + self.grid.rdy2,
                    self.rdt,
                );

                self.wp2.psi_grid[jgrid] = -(pxy * self.wp1.psi_grid[jgrid]
                    - self.grid.rdx2
                        * (self.wp1.psi_grid[jgrid - self.grid.height]
                            + self.wp2.psi_grid[jgrid + self.grid.height])
                    - self.grid.rdy2
                        * (self.wp1.psi_grid[jgrid - 1] + self.wp2.psi_grid[jgrid + 1]))
                    / pxy.conj();
            }
        }

        for jx in 0..self.grid.width {
            for jy in 0..self.grid.height {
                let val = self.wp2.psi_grid[jx * self.grid.height + jy];
                if val.norm_sqr() > self.wp2.psi_max {
                    self.wp2.psi_max = val.norm_sqr()
                };
            }
        }

        self.wp1 = self.wp2.clone();
        self.wp1.norm = self.wp1.get_norm(&self.grid);
    }

}

#[macroquad::main(window_conf)]
async fn main() {
    loop {
        //file for average data vs time
    let fl_name = format!("Wavepacket_data_{}.dat", NO);
    let mut my_file = fs::File::create(fl_name).expect("Error creating file");

    let params = Params::create();
    let mut gs = GlobalState::create(&params);
    gs.pot.plot_2d(&gs.grid);

    let fl_name = format!("{}_{}.png", "potential", NO);
    let texture_pot: Texture2D = load_texture(&fl_name).await.unwrap();

    for jt in 0..(gs.nt + 1) {
        gs.update();
        writeln!(my_file, "{} {}", jt, gs.wp1.norm).expect("Error writing to file");

        if jt % 2 == 0 {
            let fl_name = format!("{}_{}-{}.png", "wavepacket", NO, jt);
            let image = gs.wp1.plot_2d(&gs.grid);
            image.export_png(&fl_name);
            let texture = Texture2D::from_image(&image);
            draw_texture(
                texture,
                screen_width() / 2. - texture.width() / 2.,
                screen_height() / 2. - texture.height() / 2.,
                WHITE,
            );
            draw_texture(
                texture_pot,
                screen_width() / 2. - texture_pot.width() / 2.,
                screen_height() / 2. - texture_pot.height() / 2.,
                WHITE,
            );
            next_frame().await
        }
    }
    }
}

fn palette(f: f64, f_min: f64, f_max: f64) -> macroquad::color::Color {
    let z = ((f - f_min) / (f_max - f_min) * 99.0) as usize;

    macroquad::color::Color::from_rgba(
        PALETTE_2[z][0] as u8,
        PALETTE_2[z][1] as u8,
        PALETTE_2[z][2] as u8,
        255,
    )
}

const _PALETTE_1: [[u8; 3]; 100] = [
    [0, 53, 96],
    [0, 58, 102],
    [0, 63, 108],
    [0, 69, 115],
    [0, 74, 122],
    [0, 80, 130],
    [0, 86, 138],
    [0, 92, 146],
    [0, 98, 154],
    [0, 104, 162],
    [0, 110, 170],
    [0, 115, 176],
    [0, 119, 178],
    [0, 124, 180],
    [0, 128, 183],
    [0, 133, 185],
    [0, 137, 187],
    [0, 141, 190],
    [0, 146, 193],
    [0, 150, 195],
    [41, 155, 198],
    [63, 159, 200],
    [79, 164, 203],
    [92, 168, 205],
    [103, 172, 208],
    [114, 177, 210],
    [124, 181, 212],
    [133, 185, 215],
    [141, 189, 217],
    [150, 193, 219],
    [157, 197, 221],
    [165, 201, 224],
    [172, 205, 226],
    [179, 208, 228],
    [185, 212, 230],
    [192, 215, 232],
    [198, 219, 233],
    [203, 222, 235],
    [209, 225, 237],
    [214, 228, 238],
    [219, 231, 240],
    [224, 234, 241],
    [228, 236, 243],
    [232, 239, 244],
    [236, 241, 245],
    [239, 243, 246],
    [243, 245, 247],
    [245, 247, 248],
    [247, 248, 249],
    [249, 249, 249],
    [249, 248, 248],
    [248, 245, 245],
    [247, 242, 241],
    [246, 238, 237],
    [245, 234, 233],
    [244, 230, 229],
    [244, 226, 224],
    [243, 221, 219],
    [242, 217, 214],
    [241, 212, 209],
    [241, 207, 203],
    [240, 202, 198],
    [239, 197, 192],
    [238, 192, 186],
    [237, 187, 180],
    [236, 182, 174],
    [235, 176, 168],
    [234, 171, 162],
    [233, 166, 156],
    [231, 160, 149],
    [230, 154, 143],
    [229, 149, 136],
    [227, 143, 129],
    [225, 137, 122],
    [224, 131, 115],
    [222, 126, 108],
    [220, 120, 100],
    [218, 113, 92],
    [215, 107, 84],
    [213, 101, 76],
    [211, 95, 66],
    [208, 88, 56],
    [205, 82, 44],
    [202, 75, 29],
    [199, 68, 3],
    [196, 61, 0],
    [193, 53, 0],
    [186, 49, 0],
    [179, 47, 0],
    [171, 44, 0],
    [164, 41, 0],
    [156, 39, 0],
    [149, 36, 0],
    [141, 34, 0],
    [134, 31, 0],
    [126, 29, 0],
    [119, 26, 0],
    [112, 24, 0],
    [105, 22, 0],
    [97, 19, 0],
];

const PALETTE_2: [[u8; 3]; 100] = [
    [0, 0, 4],
    [1, 1, 7],
    [2, 2, 12],
    [4, 3, 17],
    [5, 4, 24],
    [8, 5, 29],
    [10, 7, 35],
    [13, 8, 41],
    [17, 10, 47],
    [20, 11, 53],
    [24, 12, 59],
    [27, 12, 66],
    [31, 12, 72],
    [35, 12, 77],
    [40, 11, 83],
    [44, 11, 88],
    [49, 10, 92],
    [54, 9, 97],
    [58, 9, 99],
    [62, 9, 102],
    [67, 10, 104],
    [71, 11, 106],
    [75, 12, 107],
    [80, 13, 108],
    [84, 15, 109],
    [88, 16, 110],
    [92, 18, 110],
    [96, 19, 110],
    [100, 21, 110],
    [105, 22, 110],
    [108, 24, 110],
    [113, 25, 110],
    [116, 26, 110],
    [120, 28, 109],
    [125, 30, 109],
    [128, 31, 108],
    [133, 33, 107],
    [136, 34, 106],
    [141, 35, 105],
    [145, 37, 104],
    [149, 38, 103],
    [153, 39, 102],
    [157, 41, 100],
    [162, 43, 98],
    [165, 44, 96],
    [169, 46, 94],
    [173, 48, 92],
    [177, 50, 90],
    [181, 52, 88],
    [185, 53, 86],
    [189, 56, 83],
    [192, 58, 81],
    [196, 60, 78],
    [199, 63, 75],
    [203, 65, 73],
    [207, 68, 70],
    [210, 70, 68],
    [213, 74, 65],
    [216, 76, 62],
    [219, 80, 59],
    [223, 82, 55],
    [225, 86, 53],
    [228, 90, 50],
    [230, 93, 47],
    [233, 97, 43],
    [235, 101, 41],
    [237, 105, 37],
    [239, 109, 34],
    [241, 113, 31],
    [243, 117, 27],
    [244, 121, 24],
    [246, 126, 20],
    [247, 131, 17],
    [248, 135, 14],
    [249, 140, 10],
    [250, 144, 8],
    [251, 150, 6],
    [251, 154, 6],
    [252, 159, 7],
    [252, 164, 9],
    [252, 168, 13],
    [252, 173, 18],
    [252, 178, 22],
    [251, 184, 28],
    [251, 189, 34],
    [250, 194, 40],
    [249, 199, 46],
    [249, 203, 53],
    [247, 208, 60],
    [246, 213, 68],
    [245, 219, 75],
    [244, 224, 84],
    [243, 229, 93],
    [242, 233, 103],
    [241, 237, 113],
    [242, 242, 124],
    [243, 245, 135],
    [245, 249, 145],
    [249, 251, 155],
    [252, 255, 164],
];
