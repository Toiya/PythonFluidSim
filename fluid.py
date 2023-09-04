import numpy as np
import pygame

### SIMULATION SETTINGS ###
# Side length of the simulation NxN square
N = 64
# Number of iterations
ITER = 4
# Scale
SCALE = 12

CANDLEMODE = True
CANDLEAMT = 100
RAINBOWMODE = False
RAINBOWRATE = 0.005

### COLORS ###
PURPLE = (180, 20, 255)
GRAY = (100, 100, 100)
RED = (50, 5, 5)
ORANGE = (200, 100, 50)
CANDLECOLOR = (150, 150, 150)
WICKCOLOR = (20, 20, 20)
RAINBOWSTARTCOLOR = (255, 0, 0)

VELOCITYSCALE = 1 if not CANDLEMODE else 3
DENSITYAMT = 200 if not CANDLEMODE else CANDLEAMT
DYECOLOR = GRAY if not RAINBOWMODE else RAINBOWSTARTCOLOR
FADERATE = 0.02

NUM_CELLS_X = NUM_CELLS_Y = N
NUM_CELLS = NUM_CELLS_X * NUM_CELLS_Y

CELL_WIDTH = CELL_HEIGHT = SCALE

SCREENWIDTH = SCALE * NUM_CELLS_X
SCREENHEIGHT = SCALE * NUM_CELLS_Y

SRC_X = SCALE * int((SCREENWIDTH/SCALE)/2)
SRC_Y = (SCREENHEIGHT - (SCALE * 15 + SCALE)) - (SCALE * 3)

GRAVITY = 0

class Fluid:
    def __init__(self, dt, diffusion, viscosity):
        self.size = N
        self.dt = dt
        self.diff = diffusion
        self.visc = viscosity

        self.s = np.zeros(N * N)
        self.density = np.zeros(N * N)

        self.Vx = np.zeros(N * N)
        self.Vy = np.zeros(N * N)

        self.Vx0 = np.zeros(N * N)
        self.Vy0 = np.zeros(N * N)

    def addDensity(self, x, y, amount):
        index = IX(x, y)
        self.density[index] += amount
    
    def addVelocity(self, x, y, amountX, amountY):
        index = IX(x, y)
        self.Vx[index] += amountX
        self.Vy[index] += (amountY - GRAVITY)

    def step(self):
        N = self.size
        visc = self.visc
        diff = self.diff
        dt = self.dt
        Vx = self.Vx
        Vy = self.Vy
        Vx0 = self.Vx0
        Vy0 = self.Vy0
        s = self.s
        density = self.density
        
        diffuse(1, Vx0, Vx, visc, dt)
        diffuse(2, Vy0, Vy, visc, dt)
        
        project(Vx0, Vy0, Vx, Vy)
        
        advect(1, Vx, Vx0, Vx0, Vy0, dt)
        advect(2, Vy, Vy0, Vx0, Vy0, dt)

        # advect_broken(1, Vx, Vx0, Vx0, Vy0, dt)
        # advect_broken(2, Vy, Vy0, Vx0, Vy0, dt)
        
        project(Vx, Vy, Vx0, Vy0)
        
        diffuse(0, s, density, diff, dt)
        advect(0, density, s, Vx, Vy, dt)
        # advect_broken(0, density, s, Vx, Vy, dt)
    
    def renderDensity(self, screen, color):
        pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, 0, SCREENWIDTH, SCREENHEIGHT))
        for i in range(N):
            for j in range(N):
                x = i * SCALE
                y = j * SCALE

                d = self.density[IX(i, j)]

                rect = pygame.Surface((SCALE, SCALE))
                rect.set_alpha(densityToAlpha(d))

                rect.fill(color)
                screen.blit(rect, (x, y))
    
    def fadeDensity(self):
        for i in range(N):
            for j in range(N):
                d = self.density[IX(i, j)]
                self.density[IX(i, j)] = d - FADERATE

# Get index from 1D array by providing x, y coords
def IX(x, y):
    x = constrain(x, 0, N - 1)
    y = constrain(y, 0, N - 1)
    return x + y * N

def constrain(val, min, max):
    if val > max: val = max
    elif val < min: val = min

    return val

def densityToAlpha(d):
    d = max(1, min(d, 255))
    # Map the density to the alpha range (0-255)
    alpha = int((d - 1) / 254 * 255)
    return alpha

def set_bnd(b, x):
    for i in range(1, N - 1):
        x[IX(i, 0)] = -x[IX(i, 1)] if b == 2 else x[IX(i, 1)]
        x[IX(i, N - 1)] = -x[IX(i, N - 2)] if b == 2 else x[IX(i, N - 2)]
        
    for j in range(1, N - 1):
        x[IX(0, j)] = -x[IX(1, j)] if b == 1 else x[IX(1, j)]
        x[IX(N - 1, j)] = -x[IX(N - 2, j)] if b == 1 else x[IX(N - 2, j)]

    x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)])
    x[IX(0, N - 1)] = 0.5 * (x[IX(1, N - 1)] + x[IX(0, N - 2)])
    x[IX(N - 1, 0)] = 0.5 * (x[IX(N - 2, 0)] + x[IX(N - 1, 1)])
    x[IX(N - 1, N - 1)] = 0.5 * (x[IX(N - 2, N - 1)] + x[IX(N - 1, N - 2)])

def diffuse(b, x, x0, diff, dt):
    a = dt * diff * (N - 2) * (N - 2)
    lin_solve(b, x, x0, a, 1 + 6 * a)

def lin_solve(b, x, x0, a, c):
    cRecip = 1.0 / c
    for _ in range(ITER):
        for j in range(1, N - 1):
            for i in range(1, N - 1):
                x[IX(i, j)] = \
                    (x0[IX(i, j)] + \
                    a * (\
                        x[IX(i + 1, j)] + \
                        x[IX(i - 1, j)] + \
                        x[IX(i, j + 1)] + \
                        x[IX(i, j - 1)])\
                    ) * cRecip
    
        set_bnd(b, x)

def project(velocX, velocY, p, div):
    for j in range(1, N - 1):
        for i in range(1, N - 1):
            div[IX(i, j)] = \
                (-0.5 * (\
                    velocX[IX(i + 1, j)] - \
                    velocX[IX(i - 1, j)] + \
                    velocY[IX(i, j + 1)] - \
                    velocY[IX(i, j - 1)]) \
                )/N
            p[IX(i, j)] = 0

    set_bnd(0, div)
    set_bnd(0, p)
    lin_solve(0, p, div, 1, 6)

    for j in range(1, N - 1):
        for i in range(1, N - 1):
            velocX[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) * N
            velocY[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) * N

    set_bnd(1, velocX)
    set_bnd(2, velocY)

# For demonstration purposes only: DO NOT USE
def advect_broken(b, d, d0, velocX, velocY, dt):
    i0 = 0
    i1 = 0
    j0 = 0
    j1 = 0
    
    dtx = dt * (N - 2)
    dty = dt * (N - 2)
    
    s0 = 0
    s1 = 0
    t0 = 0
    t1 = 0
    
    tmp1 = 0
    tmp2 = 0
    tmp3 = 0
    x = 0
    y = 0
    
    Nfloat = N - 2
    i = 0
    j = 0
    k = 0

    jfloat = 1
    ifloat = 1

    for j in range(1, N - 1):
        for i in range(1, N - 1):
            tmp1 = dtx * velocX[IX(i, j)]
            tmp2 = dty * velocY[IX(i, j)]
            x = ifloat - tmp1
            y = jfloat - tmp2

            if (x < 0.5): x = 0.5
            if (x > Nfloat + 0.5): x = Nfloat + 0.5
            i0 = np.floor(x)
            i1 = i0 + 1.0
            if (y < 0.5): y = 0.5
            if (y > Nfloat + 0.5): y = Nfloat + 0.5
            j0 = np.floor(y)
            j1 = j0 + 1.0

            s1 = x - i0
            s0 = 1.0 - s1
            t1 = y - j0
            t0 = 1.0 - t1

            i0i = int(i0)
            i1i = int(i1)
            j0i = int(j0)
            j1i = int(j1)

            d[IX(i, j)] = \
                s0 * (t0 * d0[IX(i0i, j0i)] + t1 * d0[IX(i0i, j1i)]) +\
                s1 * (t0 * d0[IX(i1i, j0i)] + t1 * d0[IX(i1i, j1i)])
            
            ifloat += 1
        jfloat += 1

    set_bnd(b, d)

def advect(b, d, d0, velocX, velocY, dt):
    u = velocX
    v = velocY
    dt0 = dt * N
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            x = i - dt0 * u[IX(i, j)]
            y = j - dt0 * v[IX(i, j)]

            if x < 0.5: x = 0.5
            if x > N + 0.5: x = N + 0.5

            i0 = int(x)
            i1 = i0 + 1

            if y < 0.5: y = 0.5
            if y > N + 0.5: y = N + 0.5

            j0 = int(y)
            j1 = j0 + 1

            s1 = x - i0
            s0 = 1 - s1

            t1 = y - j0
            t0 = 1 - t1

            d[IX(i, j)] = \
                s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) + \
                s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)])
    
    set_bnd(b, d)

def getClickedPos():
    x_pos, y_pos = pygame.mouse.get_pos()
    return x_pos, y_pos

def drawCandle(screen):
    candle_width = SCALE * 7
    candle_height = SCALE * 15

    candle_x = SCALE * int((SCREENWIDTH/SCALE)/2) - SCALE * int((candle_width/SCALE)/2)
    candle_y = SCREENHEIGHT - candle_height
    
    wick_width = SCALE
    wick_height = SCALE * 3

    wick_x = SCALE * int((SCREENWIDTH/SCALE)/2) - SCALE * int((wick_width/SCALE)/2)
    wick_y = candle_y - wick_height

    wick_tip_width = wick_width
    wick_tip_height = wick_width

    wick_tip_x = wick_x
    wick_tip_y = wick_y

    candle = pygame.Surface((candle_width, candle_height))
    candle.fill(CANDLECOLOR)

    wick = pygame.Surface((wick_width, wick_height))
    wick.fill(WICKCOLOR)

    wick_tip = pygame.Surface((wick_tip_width, wick_tip_height))
    wick_tip.fill(RED)

    screen.blit(candle, (candle_x, candle_y))
    screen.blit(wick, (wick_x, wick_y))
    screen.blit(wick_tip, (wick_tip_x, wick_tip_y))

# Chat-GPT helped me with this one
def rgb2hsl(r, g, b):
    r /= 255.0
    g /= 255.0
    b /= 255.0

    max_color = max(r, g, b)
    min_color = min(r, g, b)
    diff = max_color - min_color

    l = (max_color + min_color) / 2.0

    if diff == 0:
        h = 0
        s = 0
    else:
        if l < 0.5:
            s = diff / (max_color + min_color)
        else:
            s = diff / (2.0 - max_color - min_color)

        if max_color == r:
            h = (g - b) / diff + (6 if g < b else 0)
        elif max_color == g:
            h = (b - r) / diff + 2
        else:
            h = (r - g) / diff + 4
        h /= 6.0

    return h, s, l

# Chat-GPT helped me with this one
def hsl2rgb(h, s, l):
    if s == 0:
        r = g = b = l
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue2rgb(p, q, h + 1 / 3)
        g = hue2rgb(p, q, h)
        b = hue2rgb(p, q, h - 1 / 3)

    return int(r * 255), int(g * 255), int(b * 255)

# Chat-GPT helped me with this one
def hue2rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1 / 6:
                return p + (q - p) * 6 * t
            if t < 1 / 2:
                return q
            if t < 2 / 3:
                return p + (q - p) * (2 / 3 - t) * 6
            return p

# Chat-GPT helped me with this one
def updateColor(color):
    h, s, l = rgb2hsl(*color)
    h = (h + RAINBOWRATE) % 1.0  # Increment hue by a small amount and wrap around
    r, g, b = hsl2rgb(h, s, l)
    return r, g, b

if __name__ == "__main__":
    pygame.init()
    logo = pygame.image.load("icon.png")
    pygame.display.set_icon(logo)
    pygame.display.set_caption("Fluid Simulation")
    screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))

    fluid = Fluid(0.2, 0, 0.0000001)
    
    if CANDLEMODE: drawCandle(screen)
    
    running = True

    mousebuttondown = False
    color = DYECOLOR

    if CANDLEMODE:
        fluid.addDensity(int(SRC_X/SCALE), int((SRC_Y + 5)/SCALE), DENSITYAMT)
        amtX = np.random.uniform(0.0, 1.0) * VELOCITYSCALE
        amtY = np.random.uniform(0.0, 1.0) * VELOCITYSCALE
        fluid.addVelocity(int(SRC_X/SCALE), int((SRC_Y + 5)/SCALE), amtX, amtY)
    
    (pmouseX, pmouseY) = pygame.mouse.get_pos()

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mousebuttondown = True
            elif event.type == pygame.MOUSEBUTTONUP:
                mousebuttondown = False
            
        if mousebuttondown == True:
            x_pos, y_pos = getClickedPos()
            
            d = -(CANDLEAMT/3) if CANDLEMODE else DENSITYAMT
            
            fluid.addDensity(int(x_pos/SCALE), int(y_pos/SCALE), d)
            amtX = (x_pos - pmouseX) * VELOCITYSCALE
            amtY = (y_pos - pmouseY) * VELOCITYSCALE
            fluid.addVelocity(int(x_pos/SCALE), int(y_pos/SCALE), amtX, amtY)

            if RAINBOWMODE: color = updateColor(color)
        
        if CANDLEMODE:
            fluid.addDensity(int(SRC_X/SCALE), int(SRC_Y/SCALE), DENSITYAMT)
            amtX = np.random.uniform(-0.5, 0.5) * VELOCITYSCALE
            amtY = np.random.uniform(-1.0, 0.0) * VELOCITYSCALE
            fluid.addVelocity(int(SRC_X/SCALE), int(SRC_Y/SCALE), amtX, amtY)

        fluid.step()
        fluid.renderDensity(screen, color)
        fluid.fadeDensity()

        if CANDLEMODE: drawCandle(screen)

        pygame.display.flip()

        (pmouseX, pmouseY) = pygame.mouse.get_pos()