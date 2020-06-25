"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding



from gym.envs.registration import register

#woot
register(
    id='MountTest-v0',
    entry_point='mountain_car:MountainCarEnv',
)





class MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.h=1
        
        #Make them into boxes later         
        self.initial_conditions=np.random.random(9)
        self.default_conditions=np.asarray([0]*8+[1])

        
        
        self.force=0.001*0.3
        self.gravity=0.0025*0.3

        self.low = np.array([-self.max_speed]+[-2.5]*12)
        self.high = np.array([self.max_speed]+[2.5]*12)
        print(self.low,self.high)
        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.seed()
        self.reset()
        
        
    def hill(self,k):
        """Defines the height of the hill at position k
        This is just for rendering and should be an integral of slope
"""
        x = self.initial_conditions
        k1=k+1.5
        sum_= -x[8]*np.exp(-k1**2)+x[8]
        for i in range(4):
            k2=(k+3-x[i+4]*3)
            sum_+= (x[i]-0.5)*np.exp(-(i+1)*k2**2)
        return sum_

    def slope(self,k):
        """Defines the slope of the hill at position k"""
        x = self.initial_conditions
        k1=k+1.5
        sum_= 2*k1*x[8]*np.exp(-k1**2)
        for i in range(4):
            k2=(k+3-x[i+4]*3)
            sum_+= -2*(i+1)*k2*(x[i]-0.5)*np.exp(-(i+1)*k2**2)
        return sum_
    
    def nearby(self):
        xs=np.linspace(self.pos-0.4,self.pos+0.4,12)
        ys = self._height(xs)
        return xs,ys
    def set_initials(self,action):
        self.initial_conditions=action
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        velocity = self.state[0]
        #print((action-1)*self.force,self.slope(3*position)*(-self.gravity))
        velocity += (action-1)*self.force + self.slope(3*self.pos)*(-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        self.pos += velocity
        self.pos = np.clip(self.pos, self.min_position, self.max_position)
        if (self.pos==self.min_position and velocity<0): velocity = 0

        self.timesteps+=1
        done = bool(self.pos >= self.goal_position or self.timesteps>199)
        reward = -1.0

        self.state[0] = velocity
        #subtract current height to get relative position
        self.state[1:13]=self.nearby()[1]-self._height(self.pos)
        return np.array(self.state), reward, done, {}
    def defaults(self):
        self.initial_conditions = self.default_conditions.copy()
        self.reset()
    
    def reset(self):
        #self.initial_conditions=np.random.random(4)
        self.state = np.zeros(13)
        self.pos=self.np_random.uniform(low=-0.6, high=-0.4)
        self.dft=np.fft.fft(self.initial_conditions)
        self.timesteps=0
        self.state[1:13]=self.nearby()[1]
        return np.array(self.state)

    def _height(self, xs):
        return self.hill(3 * xs)#*.45*self.h+.55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width*0.6
        carwidth=40
        carheight=20
        off=[10,150]
        
        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        off[1]=-min(ys)*scale+10
        xys = list(zip((xs-self.min_position)*scale+off[0], ys*scale+off[1]))

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.r=rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)


            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)



        self.track = self.r.make_polyline(xys)
        self.track.set_linewidth(2)
        self.viewer.add_onetime(self.track)

        
        flagx = (self.goal_position-self.min_position)*scale+off[0]
        flagy1 = self._height(self.goal_position)*scale+off[1]
        flagy2 = flagy1 + 50
        flagpole = self.r.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_onetime(flagpole)
        flag = self.r.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
        flag.set_color(0,.8,0)
        self.viewer.add_onetime(flag)

        xs,ys=self.nearby()
        ys-=0.01
        #print(ys)
        xys = list(zip((xs-self.min_position)*scale+off[0], ys*scale+off[1]))

        self.track = self.r.make_polyline(xys)
        self.track.set_linewidth(6)
        self.track.set_color(1.0,0.75,0)
        self.viewer.add_onetime(self.track)
        pos = self.pos
        self.cartrans.set_translation((pos-self.min_position)*scale+off[0], self._height(pos)*scale+off[1])
        #This will visually show if the slope function is correct.
        self.cartrans.set_rotation(np.arctan(self.slope(3 * pos)*3))#self.slope(3 * pos)*0.8)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    
    def get_keys_to_action(self):
        return {():1,(276,):0,(275,):2,(275,276):1} #control with left and right arrow keys 
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

##    def hill(self,k):
##        k-=0.5
##        x=self.dft
##        N=x.size
##        sum_=0
##        for n in range(N):
##            sum_+=x[n]*np.e**(-2j*np.pi*(N-k)*n/N)
##        #returning only the real bit
##        #return sum_/N
##        return sum_.real/N
##
##    def slope(self,k):
##        k-=0.5
##        x=self.dft
##        N=x.size
##        sum_=0
##        for n in range(N):
##            #not making it as steep. . . JK
##            sum_+=(2j*np.pi*n/N)*x[n]*np.e**(-2j*np.pi*(N-k)*n/N)
##            #sum_+=(1j*n)*x[n]*np.e**(-2j*np.pi*(N-k)*n/N)
##        #returning only the real bit
##        #return sum_/N
##        return sum_.real/N

