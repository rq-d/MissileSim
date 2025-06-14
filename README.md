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

--

authors: Raul Quintana

state:  prediscussion

--
![alt text](prompt/photos/missile.PNG "Title")
## Approach


### Development Path
Objective: Explore max range of a hypothetical rocket

Gather as much concrete data as possible, make assumptions for the rest. Simulate it using Elodin and test it in different conditions to find a hypothetical max range and identify contributing variables.

The rocket motor and payload mass were provided. I used the rocket diameter as a constraint for the warhead diameter to minimize the frontal area and, hopefully, reduce drag to increase speed. Using a common explosive (RDX) as a density reference, I estimated the volume for a 5 kg payload and solved for the cylinder height, which turned out to be 0.622 meters. Adding the motor length of nearly 1 meter brings the total length to about 1.6 meters—without accounting for electronics—indicating this will be a large rocket. I added 100 mm and 250 g of electronics to the SWaP budget for a sense of realism. The remaining mass would come from the airframe, which I chose to be a 4 mm carbon fiber weave tube.

Now we have enough information to calculate the total mass, moments of inertia, and center of gravity (CG) properties. Since the rocket is a simple combination of tubes and cylinders, it was faster and easier to model it as a SolidWorks assembly and extract the mass properties, rather than manually calculating them using the parallel axis theorem. In retrospect, this was a good decision, as the moment of inertia became a pain point for aerodynamic stability later on.

For simplicity, I decided to use the aerodynamic forces and moments from the smaller rocket in the Elodin template examples. I also used the vertical PID acceleration control from that simulation, along with a guidance model I had developed while preparing for this exercise.

The thrust profile from AeroTech sits in `models/data/AeroTech_M685.txt` and is parsed in `rocket.py`

Now it's time to write code. I began with a proposed directory structure to improve readability, breaking the simulation into separate files and folders. The goal is to have a top-level `main.py` that simply calls `sim.py`. The `sim.py` file creates the world, spawns entities, and selects which models to use. Models are stored in the models folder, and eventually, each model will reside in its own file.

Similarly, each entity—such as the rocket or target—will have its own file (e.g., `rocket.py`, `target.py`) containing definitions specific to that entity. I also created a `utils` folder to store commonly used functions like coordinate transformations, math operations, and parsing tools. For now, it contains just a few transformation functions. Admittedly, I did not completely follow this design pattern due to time constraints but I believe it would improve modularity, and readability.

```
MissileSim/prompt
├── main.py
├── models
│   ├── data
│   │   └── AeroTech_M685W.txt
│   ├── models.py
│   ├── rocket.py
│   └── target.py
├── monte.py
├── photos
├── plot.py
├── sim.py
└── utils
    └── math.py
```

Now that all the models are installed, I can begin testing the flight behavior. Without changing any parameters from the previously well-behaved and tuned template model, this version tumbles upon release. Since this motor produces roughly three times the thrust of the one used in the template rocket simulation, I wasn’t surprised that additional tuning would be necessary.

To run the simulation faster than real time for quicker iteration, I wrote a script called `plot.py`, which executes the scenario in headless mode and plots key data.

Now for tuning: I started by disabling the zero-acceleration setpoint and attempted to achieve dynamic stability without control input. I quickly realized that stacking the internal components of the rocket—payload, electronics, and motor—was efficient in terms of volume, but it also resulted in very low moments of inertia about the Y and Z axes. To reduce the influence of aerodynamic moments, I extended the rocket’s length significantly, ultimately settling at 2.5 meters (a photo of the rocket is at the bottom of this README). After considerable effort, I was able to drastically reduce the magnitude of the vertical pitch instability, though some instability still remains.

I tuned the pitch-axis PID controller to enable zero vertical acceleration control. This didn’t take much effort, but I did notice that the vehicle was very sensitive to angle of attack (AoA). Following this, I attempted to tune the Proportional Navigation terminal guidance model for the new vehicle but encountered significant issues with AoA instability—even with a small guidance constant. It only worked when the target was placed close enough to the rocket's current trajectory that the resulting control inputs were small and didn’t require a large change in direction, thus avoiding AoA instability. With this constraint, the guidance model wasn’t useful for this project (nor is it the main focus), so I disabled it and moved on.
P.S. — It works fantastically in a modified version of the example found in examples/rocket/main.py, if you're curious.

Lastly, I wrote a top-level script called monte.py that takes random samples of some simulation inputs, runs the sim multiple times, and generates plots. It also saves the input and output dictionaries as JSON files to make parsing easier later on. From the few tests I’ve run, launch angle seems to have the biggest impact on range, with the sweet spot being around 50 degrees from horizontal. I dropped a set of figures from 50 runs at the end of the README.

# Working Notes

### Constraints Driving the Design
#### Motor:
- mass: 7.008kg
- length: 936mm -> 0.936m
- diameter: 75mm -> 0.075m ------ this drives missile diameter

#### Warhead:
- assuming RDX explosive
- mass 5kg
- density: 1.82g/cm3
- calced volume if using motor diameter: 2747 cm3 --> 0.002747 m^3
- calced length if using motor diameter: h=V/pi*r^2 -> h=0.002747/(pi*(0.075/2)^2) -> 0.6218m

#### Body
- assume a 4mm thin walled carbon tube
- 1.5578m <- warhead + motor length
- +100mm and +250g if we add electronics (assume 4 inches of length occupied)
- diameter: 0.075m
- overall length: 1.6578m
- mass: 12.0081.1 +0.250kg = 13.450kg

#### Complete Missile:
- Modeled a 4mm carbon airframe around the rocket (motor, electronics, warhead). Adjusted length and weight distrubution to find a good Moment of Inertia
- Final Mass: 18.98kg
- Principal moments of inertia at center of mass (SW coordinate system same as body frame)
  - 0.03, 10.52, 10.52 Px, Py, Pz

#### Aero and Mass Properties
- using aero coefficients and math from example given.
- Mass properties are coming from the solidworks assembly.
- `Coordinate 2` is placed at the rocket tip with x facing the direction of flight

#### TODO
- Turn wind model on
- Explore any guidance laws that will help range
- More monte carlo inputs. Namely, Wind, small thrust misalignments, launch elevation/angles, small CG variations
- Better representation of new rocket body in Aero models
- Represent the CG location correctly.
- Had difficulty getting the target state into the pronav model(component). I believe the 3-body example is doing this by using a fold function but I dont understand how entirely. The ball example only has one physical entity (the ball itself) so i couldnt find the pattern im looking for there. I decided to move on since the focus of this prompt is to explore the range of an unguided rocket, thus I put a static target position in the model as a temporary hack. Guidance would be a fun feature to have so i left a working rocket with pronav applied to the vertical PID controller in examples/rocket/sim.py as well, run it with elodin editor examples/rocket/main.py if curious

# Run Data --- 50 MC
![alt text](prompt/photos/figure_2.png "Title")
![alt text](prompt/photos/figure_1.png "Title")
![alt text](prompt/photos/figure_3.png "Title")
![alt text](prompt/photos/figure_4.png "Title")
![alt text](prompt/photos/figure_5.png "Title")