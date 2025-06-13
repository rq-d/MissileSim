# Missile Sim
# Quickstart

Clone this branch and cd into it
```
git clone -b prompt git@github.com:rq-d/MissileSim.git
cd MissileSim
```

The `prompt` folder contains everything needed for this example. To run the visualized version of the sim. Call the file `prompt/main.py` as so. I have the elodin executable placed at the top level of this repository, so no need to change directories again.
```
./elodin editor prompt/main.py
```

I placed two more scripts in there called plot.py and monte.py. The former is used to test new features and updates, the latter is used to gather data over more runs. monte.py also saves figures and input/output values in `prompt/photos`. It's currently set up to take uniform random samples of a few variables that impact the flight path, including the launch angle.
```
python3 prompt/monte.py
```

# RFD Request For Discussion
_
authors: Raul Quintana
state:prediscussion
_

- architecture, utils folder, models folder, sim and main files
- motor from the link
- implemented a vertical guidance law in this example but the current rocket is sensitive to moderate fin deflections. Thus pronav works at times but usually causes instability. Its in rocket.py but is inactive.
- Had difficulty getting the target state into the pronav model(component). I believe the 3-body example is doing this by using a fold function. The ball example only has one physical entity (the ball itself) so i couldnt find the pattern im looking for there. I decided to move on since the focus of this prompt is to explore the range of an unguided rocket, thus I put a static target position in the model as a temporary hack. Guidance would be a fun feature to have so i left a working rocket with pronav applied to the vertical PID controller in examples/rocket/sim.py as well, run it with elodin editor examples/rocket/main.py if curious
- 

# Notes
### Motor:

- mass: 7.008kg
- length: 936mm -> 0.936m
- diameter: 75mm -> 0.075m ------ this drives missile diameter

### Warhead:

- assumed RDX explosive
- mass 5kg
- density: 1.82g/cm3
- calced volume if using motor diameter: 2747 cm3 --> 0.002747 m^3
- calced length if using motor diameter: h=V/pi*r^2 -> h=0.002747/(pi*(0.075/2)^2) -> 0.6218m

### Body

- assume a 4mm thin walled carbon tube
- 1.5578m <- warhead + motor length
- +100mm and +250g if we add electronics (assume 4 inches of length occupied)
- diameter: 0.075m
- overall length: 1.6578m
- mass: 12.0081.1 +0.250kg = 13.450kg

### Complete Missile:
Modeled a 4mm carbon airframe around the rocket (motor, electronics, warhead). Adjusted length and weight distrubution to find a good Moment of Inertia

- Final Mass: 18.98kg
- Principal moments of inertia at center of mass (SW coordinate system is rotated differently than body frame. origin at missile base)
  - 0.03, 10.52, 10.52 Px, Py, Pz

Aero and Mass Properties

- using aero coefficients and math from example given.
- Mass properties are coming from the solidworks assembly.
- `Coordinate 2` is placed at the rocket tip with x facing the direction of flight

![alt text](prompt/photos/missile.PNG "Title")


![alt text](prompt/photos/figure_2.png "Title")
![alt text](prompt/photos/figure_1.png "Title")
![alt text](prompt/photos/figure_3.png "Title")
![alt text](prompt/photos/figure_4.png "Title")
![alt text](prompt/photos/figure_5.png "Title")