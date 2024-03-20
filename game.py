"""
    Gra polega na zastrzeleniu wszystkich(9) wrogów.
    Mają oni po 2 życia przy jednym trafieniu odejmuje im się po jednym życiu.
    Kiedy zginą opadają na dno planszy.
    Gra po skończeniu nie włącza się ponownie.
"""

import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr

class SimpleComponent:

    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)

    def update(self, rate):
        self.position += rate * self.velocity


class SentientComponent:

    def __init__(self, position, eulers, health):
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.velocity = np.array([0, 0, 0], dtype=np.float32)
        self.state = "fallingOn"
        self.health = health
        self.canShoot = True
        self.reloadTime = 0
        self.fallTime = 0

    def update(self, rate):

        if self.state == "stable":

            self.position += self.velocity * rate

            self.position[1] = min(2, max(-2, self.position[1]))

            if not self.canShoot:
                self.reloadTime -= rate
                if self.reloadTime < 0:
                    self.reloadTime = 0
                    self.canShoot = True

            if self.health < 0:
                self.state = "fallingOff"

        elif self.state == "fallingOn":
            self.position[2] = (0.9 ** self.fallTime) * 2
            self.fallTime += rate
            if self.position[2] < 1:
                self.fallTime = 0
                self.position[2] = -1
                self.state = "stable"

        else:
            self.position[2] = -8 + (0.9 ** self.fallTime) * 9
            self.fallTime += rate


class Scene:

    def __init__(self):
        self.enemySpawnRate = 0.02

        self.powerupSpawnRate = 0.01

        self.enemyShootRate = 0.02

        self.player = SentientComponent(
            position=[3, 0.3, -1],
            eulers=[0, 90, 0],
            health=12
        )

        self.xmin = 7
        self.xmax = 15
        self.ymin = -1
        self.ymax = 1

        self.enemies = []

        self.bullets = []

        self.powerups = []

    def update(self, rate):

        if np.random.uniform() < self.enemySpawnRate * rate and len(self.enemies) < 9:
            newEnemy = SentientComponent(
                position=[
                    np.random.uniform(low=self.xmin, high=self.xmax),
                    np.random.uniform(low=self.ymin, high=self.ymax),
                    -1
                ],
                eulers=[0, 0, 0],
                health=2
            )
            newEnemy.velocity[1] = np.random.uniform(low=-0.1, high=0.1)
            self.enemies.append(newEnemy)

        self.player.update(rate)
        self.player.velocity = np.array([0, 0, 0], dtype=np.float32)


        for bullet in self.bullets:
            bullet.update(rate)
            if bullet.position[0] > 48 or bullet.position[0] < -48:
                self.bullets.pop(self.bullets.index(bullet))

            for enemy in self.enemies:
                if bullet.position[0] > enemy.position[0] - 0.1 \
                    and bullet.position[0] < enemy.position[0] + 0.2 \
                    and bullet.position[1] > enemy.position[1] - 0.1 \
                    and bullet.position[1] < enemy.position[1] + 0.2:

                        enemy.health -= 1
                        break


        for enemy in self.enemies:
            enemy.update(rate)

            if enemy.position[1] >= self.ymax or enemy.position[1] <= self.ymin:
                enemy.velocity[1] *= -1

    def move_player(self, dPos):

        self.player.velocity = dPos

    def player_shoot(self):

        if self.player.canShoot:
            self.bullets.append(
                SimpleComponent(
                    position=[7, self.player.position[1], -1],
                    velocity=[3, 0, 0]
                )
            )
            self.player.canShoot = False
            self.player.reloadTime = 15


class App:

    def __init__(self, screenWidth, screenHeight):

        self.screenWidth = screenWidth
        self.screenHeight = screenHeight

        self.renderer = GraphicsEngine()

        self.scene = Scene()

        self.lastTime = pg.time.get_ticks()
        self.currentTime = 0
        self.numFrames = 0
        self.frameTime = 0
        self.lightCount = 0

        self.mainLoop()

    def mainLoop(self):
        running = True
        while (running):
            # check events
            for event in pg.event.get():
                if (event.type == pg.QUIT):
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False

            self.handleKeys()

            self.scene.update(self.frameTime * 0.05)

            self.renderer.render(self.scene)

            # timing
            self.calculateFramerate()
        self.quit()

    def handleKeys(self):

        keys = pg.key.get_pressed()
        rate = self.frameTime / 16

        # left, right, space
        if keys[pg.K_LEFT]:
            self.scene.move_player(rate * np.array([0, 1, 0], dtype=float))
        elif keys[pg.K_RIGHT]:
            self.scene.move_player(rate * np.array([0, -1, 0], dtype=float))

        if keys[pg.K_SPACE]:
            self.scene.player_shoot()

    def calculateFramerate(self):

        self.currentTime = pg.time.get_ticks()
        delta = self.currentTime - self.lastTime
        if (delta >= 1000):
            framerate = max(1, int(1000.0 * self.numFrames / delta))
            pg.display.set_caption(f"Running at {framerate} fps.")
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = float(1000.0 / max(1, framerate))
        self.numFrames += 1

    def quit(self):

        self.renderer.destroy()


class GraphicsEngine:

    def __init__(self):
        self.palette = {
            "Red": np.array([255 / 255, 93 / 255, 93 / 255], dtype=np.float32),
            "Green": np.array([93 / 255, 255 / 255, 93 / 255], dtype=np.float32),
            "LightBlue": np.array([166 / 255, 227 / 255, 233 / 255], dtype=np.float32),
            "Dark": np.array([27 / 255, 36 / 255, 48 / 255], dtype=np.float32),
            "Coffee": np.array([213 / 255, 206 / 255, 163 / 255], dtype=np.float32),
            "Gray": np.array([127 / 255, 132 / 255, 135 / 255], dtype=np.float32),
            "White": np.array([255 / 255, 255 / 255, 255 / 255], dtype=np.float32),
        }

        # initialise pygame
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK,
                                    pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode((640, 480), pg.OPENGL | pg.DOUBLEBUF)

        # initialise opengl
        glClearColor(self.palette["LightBlue"][0], self.palette["LightBlue"][1], self.palette["LightBlue"][2], 1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # create renderpasses and resources
        shader = self.createShader("shaders/vertex.txt", "shaders/fragment.txt")
        self.renderPass = RenderPass(shader)
        self.blockMesh = Mesh("models/block.obj")
        self.planeMesh = Mesh("models/plane.obj")
        self.windowMesh = Mesh("models/window.obj")
        self.playerMesh = Mesh("models/player.obj")
        self.bulletMesh = Mesh("models/bullet.obj")
        self.enemiesMesh = Mesh("models/player.obj")

    def createShader(self, vertexFilepath, fragmentFilepath):
        with open(vertexFilepath, 'r') as f:
            vertex_src = f.readlines()

        with open(fragmentFilepath, 'r') as f:
            fragment_src = f.readlines()

        shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                compileShader(fragment_src, GL_FRAGMENT_SHADER))

        return shader

    def render(self, scene):
        # refresh screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.renderPass.render(scene, self)

        pg.display.flip()

    def destroy(self):
        pg.quit()


class RenderPass:

    def __init__(self, shader):
        # initialise opengl
        self.shader = shader
        glUseProgram(self.shader)

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy=45, aspect=800 / 600,
            near=0.1, far=50, dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"),
            1, GL_FALSE, projection_transform
        )
        self.modelMatrixLocation = glGetUniformLocation(self.shader, "model")
        self.viewMatrixLocation = glGetUniformLocation(self.shader, "view")
        self.colorLoc = glGetUniformLocation(self.shader, "object_color")

    def render(self, scene, engine):
        glUseProgram(self.shader)

        view_transform = pyrr.matrix44.create_look_at(
            eye=np.array([-0.6, 0, 0.3], dtype=np.float32),
            target=np.array([2, 0, 0], dtype=np.float32),
            up=np.array([0, 0, 1], dtype=np.float32)
        )
        glUniformMatrix4fv(self.viewMatrixLocation, 1, GL_FALSE, view_transform)

        # town
        modelTransform = pyrr.matrix44.create_identity(dtype=np.float32)
        modelTransform = pyrr.matrix44.multiply(
            m1=modelTransform,
            m2=pyrr.matrix44.create_from_z_rotation(theta=np.radians(180), dtype=np.float32)
        )
        modelTransform = pyrr.matrix44.multiply(
            m1=modelTransform,
            m2=pyrr.matrix44.create_from_x_rotation(theta=np.radians(90), dtype=np.float32)
        )
        modelTransform = pyrr.matrix44.multiply(
            m1=modelTransform,
            m2=pyrr.matrix44.create_from_translation(vec=np.array([12, 1.5, -1], dtype=np.float32))
        )
        glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, modelTransform)
        glUniform3fv(self.colorLoc, 1, engine.palette["Coffee"])
        glBindVertexArray(engine.blockMesh.vao)
        glDrawArrays(GL_TRIANGLES, 0, engine.blockMesh.vertex_count)
        glUniform3fv(self.colorLoc, 1, engine.palette["Dark"])
        glBindVertexArray(engine.planeMesh.vao)
        glDrawArrays(GL_TRIANGLES, 0, engine.planeMesh.vertex_count)
        glUniform3fv(self.colorLoc, 1, engine.palette["Gray"])
        glBindVertexArray(engine.windowMesh.vao)
        glDrawArrays(GL_TRIANGLES, 0, engine.windowMesh.vertex_count)


        # player
        glUniform3fv(self.colorLoc, 1, engine.palette["Green"])
        modelTransform = pyrr.matrix44.create_identity(dtype=np.float32)
        modelTransform = pyrr.matrix44.multiply(
            m1=modelTransform,
            m2=pyrr.matrix44.create_from_x_rotation(theta=np.radians(270), dtype=np.float32)
        )
        modelTransform = pyrr.matrix44.multiply(
            m1=modelTransform,
            m2=pyrr.matrix44.create_from_z_rotation(theta=np.radians(270), dtype=np.float32)
        )
        modelTransform = pyrr.matrix44.multiply(
            m1=modelTransform,
            m2=pyrr.matrix44.create_from_scale(scale=np.array([0.15, 0.15, 0.15]), dtype=np.float32)
        )
        modelTransform = pyrr.matrix44.multiply(
            m1=modelTransform,
            m2=pyrr.matrix44.create_from_translation(vec=scene.player.position, dtype=np.float32)
        )
        glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, modelTransform)
        glBindVertexArray(engine.playerMesh.vao)
        glDrawArrays(GL_TRIANGLES, 0, engine.playerMesh.vertex_count)

        # bullet
        glUniform3fv(self.colorLoc, 1, engine.palette["White"])
        for bullet in scene.bullets:
            modelTransform = pyrr.matrix44.create_identity(dtype=np.float32)
            modelTransform = pyrr.matrix44.multiply(
                m1=modelTransform,
                m2=pyrr.matrix44.create_from_scale(scale=np.array([0.10, 0.10, 0.10]), dtype=np.float32)
            )
            modelTransform = pyrr.matrix44.multiply(
                m1=modelTransform,
                m2=pyrr.matrix44.create_from_translation(vec=bullet.position, dtype=np.float32)
            )
            glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, modelTransform)
            glBindVertexArray(engine.bulletMesh.vao)
            glDrawArrays(GL_TRIANGLES, 0, engine.bulletMesh.vertex_count)

        # enemies
        for enemy in scene.enemies:
            glUniform3fv(self.colorLoc, 1, engine.palette["Red"])
            modelTransform = pyrr.matrix44.create_identity(dtype=np.float32)
            modelTransform = pyrr.matrix44.multiply(
                m1=modelTransform,
                m2=pyrr.matrix44.create_from_x_rotation(theta=np.radians(270), dtype=np.float32)
            )
            modelTransform = pyrr.matrix44.multiply(
                m1=modelTransform,
                m2=pyrr.matrix44.create_from_z_rotation(theta=np.radians(90), dtype=np.float32)
            )
            modelTransform = pyrr.matrix44.multiply(
                m1=modelTransform,
                m2=pyrr.matrix44.create_from_scale(scale=np.array([0.15, 0.15, 0.15]), dtype=np.float32)
            )
            modelTransform = pyrr.matrix44.multiply(
                m1=modelTransform,
                m2=pyrr.matrix44.create_from_translation(vec=enemy.position, dtype=np.float32)
            )
            glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, modelTransform)
            glBindVertexArray(engine.enemiesMesh.vao)
            glDrawArrays(GL_TRIANGLES, 0, engine.enemiesMesh.vertex_count)

    def destroy(self):
        glDeleteProgram(self.shader)


class Mesh:

    def __init__(self, filename):
        # x, y, z, s, t, nx, ny, nz
        self.vertices = self.loadMesh(filename)
        self.vertex_count = len(self.vertices) // 3
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

    def loadMesh(self, filename):

        # raw, unassembled data
        v = []

        # final, assembled and packed result
        vertices = []

        # open the obj file and read the data
        with open(filename, 'r') as f:
            line = f.readline()
            while line:
                firstSpace = line.find(" ")
                flag = line[0:firstSpace]
                if flag == "v":
                    # vertex
                    line = line.replace("v ", "")
                    line = line.split(" ")
                    l = [float(x) for x in line]
                    v.append(l)
                elif flag == "f":
                    # face, three or more vertices in v/vt/vn form
                    line = line.replace("f ", "")
                    line = line.replace("\n", "")
                    # get the individual vertices for each line
                    line = line.split(" ")
                    faceVertices = []
                    for vertex in line:
                        # break out into [v,vt,vn],
                        # correct for 0 based indexing.
                        l = vertex.split("/")
                        position = int(l[0]) - 1
                        faceVertices.append(v[position])
                    # obj file uses triangle fan format for each face individually.
                    # unpack each face
                    triangles_in_face = len(line) - 2

                    vertex_order = []
                    """
                        eg. 0,1,2,3 unpacks to vertices: [0,1,2,0,2,3]
                    """
                    for i in range(triangles_in_face):
                        vertex_order.append(0)
                        vertex_order.append(i + 1)
                        vertex_order.append(i + 2)
                    for i in vertex_order:
                        for x in faceVertices[i]:
                            vertices.append(x)
                line = f.readline()
        return vertices

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))


myApp = App(800, 600)