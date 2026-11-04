Now that your MuJoCo arm and cube simulation is running, you're ready for the part that actually interests me most about your project: getting a neural network to learn the physics of its environment through experience.

I'd approach this as a series of controlled experiments, rather than immediately introducing reinforcement learning or a sophisticated world-model architecture.

Your first ML milestone should be:

Train a neural network to predict how the robot and cube will move in response to the robot's actions, using only previously collected experience.

Then test whether it can predict interactions it has never seen before.

## 1. The architecture of your first learning system

MuJoCo simulation

Ground-truth physics environment

Experience collection

Record observations, actions, next observations

PyTorch neural network

Learn the relationship between actions and physical outcomes

Prediction evaluation

Compare predicted states against unseen MuJoCo trajectories

Future: Model-based planning

Choose actions by predicting their consequences

The key is that MuJoCo provides the physics, but the neural network doesn't have access to MuJoCo's physics calculations. It only sees observations and actions.

## 2. Define what the network learns

For your two-joint arm, I'd start with a numerical state representation.

| Information           | Variables                        | Dimensions |
| --------------------- | -------------------------------- | ---------- |
| Joint positions       | \\(q_1,q_2\\)                    | 2          |
| Joint velocities      | \\(\dot q_1,\dot q_2\\)          | 2          |
| Cube position         | \\(x,y,z\\)                      | 3          |
| Cube orientation      | Quaternion                       | 4          |
| Cube linear velocity  | \\(v_x,v_y,v_z\\)                | 3          |
| Cube angular velocity | \\(\omega_x,\omega_y,\omega_z\\) | 3          |
| Total                 |                                  | 17         |

Your actions might be two motor commands:

\\[ a_t=[u_1,u_2] \\]

So the neural network receives 19 numbers: 17 state variables and 2 action variables.

It predicts the change in state after a fixed time interval:

\\[ \boxed{\Delta\hat{s}\_t=f\_\theta(s_t,a_t)} \\]

And:

\\[ \hat{s}\_{t+1}=s_t+\Delta\hat{s}\_t \\]

For orientation, you'll want to handle quaternions separately rather than simply adding their four components. An even simpler first version is to use a planar state representation if your cube's motion is constrained to a plane.

I would use a prediction interval of approximately 50–100 ms initially, holding the action constant over that interval.

Important: If your actuators use position targets, the network needs to learn the effects of those targets, not assume the commands are applied as joint torques.

## 3. Start with a very small neural network

You don't need a transformer, an LLM, or even a GPU for this.

A simple multilayer perceptron (MLP) is sufficient.

Example neural network

Conceptual MLP diagram; node counts are illustrative.

A reasonable starting configuration:

- 3 hidden layers, 128 neurons each
- ReLU or SiLU activations
- Adam optimizer
- Mean squared prediction error
- Normalized input and output variables

This should train comfortably on your laptop's CPU for modest datasets.

## 4. Data collection is the most important part

Before training anything, you'll need to generate experience.

I would collect roughly 50,000 transitions initially, but the diversity of those transitions matters much more than the count.

The biggest trap is collecting thousands of samples where the arm moves around without touching the cube. The model could achieve a very low prediction error simply by learning that the cube usually doesn't move.

I'd deliberately collect three categories of experience:

Free movement

Move the joints through different trajectories without touching the cube. Learn basic arm dynamics.

Contact interactions

Position the cube within the arm's reachable workspace. Generate pushes, glancing contacts, and collisions at different speeds and angles.

Randomized conditions

Vary initial joint positions, cube positions, and eventually friction, mass, and other physical properties.

I'd start with approximately equal amounts of free-motion and contact-rich data, then adjust based on model performance.

You can run MuJoCo headlessly, without rendering every frame, to collect data much faster than real time.

Crucially, split your dataset by entire simulation episodes, not randomly by individual transitions. Otherwise, nearly identical consecutive states can leak between training and testing.

## 5. Evaluate whether it's actually learning physics

This is where I'd spend substantial effort.

A decreasing training loss doesn't necessarily mean your network has learned useful physical dynamics.

I'd test three capabilities.

Test A — One-step prediction

Given a state and action, how accurately can the network predict the state 100 ms later?

Test B — Multi-step prediction

Starting from an initial state, repeatedly feed the network's predicted state back into itself.

Does it predict the next second of movement reasonably well, or does the trajectory quickly diverge?

Test C — Generalization

Place the cube somewhere the model hasn't encountered in training. Does it still predict what happens when the arm pushes it?

I'd also compare against a simple no-change baseline and report errors separately for contact and non-contact interactions.

### What the results might look like

Illustrative multi-step prediction error

Hypothetical results, not measurements from your robot.

Learned dynamics modelNo-change baseline

00.0450.090.1350.180.10.20.40.60.81

The interesting result isn't just that the learned model predicts better than a trivial baseline. It's whether it remains useful over a long enough horizon to support planning.

Contact events are particularly challenging because small prediction errors can lead to completely different outcomes.

## 6. Your next concrete development task

Since Codex CLI already helped you build the simulation, I'd have it extend the existing project rather than starting a separate ML project.

I'd break the implementation into these milestones:

| Step | Deliverable                                                   |
| ---- | ------------------------------------------------------------- |
| 1    | Headless experience collector saving state-action transitions |
| 2    | Dataset inspection and train/validation/test splitting        |
| 3    | PyTorch MLP dynamics model                                    |
| 4    | Training script with loss logging                             |
| 5    | One-step and multi-step evaluation                            |
| 6    | Visual comparison of predicted versus simulated trajectories  |

You can keep everything inside VS Code, and you don't need to change your hardware.

One engineering detail worth enforcing from the beginning: have Codex document the exact observation vector, action definition, control timestep, and coordinate frames. Those definitions need to remain consistent between data collection, training, and evaluation.

## 7. Where this is ultimately going

Once the model can predict the cube's movement, the next major experiment is model-based control.

Instead of commanding random actions, you give the robot a goal: move the cube to a target position.

The robot can then use its learned model to evaluate candidate action sequences:

\\[ a^\*\_{t:t+H}=\arg\min\_{a\_{t:t+H}} J(\hat{s}\_{t+H},s\_{\text{goal}}) \\]

In plain language: try different future actions inside the learned model, predict their outcomes, and select the sequence that best achieves the goal.

That's the point where your neural network stops being merely a predictor and becomes part of an intelligent control system.

My recommendation for right now: Implement data collection first, and don't start training until you've inspected the dataset and confirmed that it contains meaningful contact interactions.

One question that will materially affect how we build the learning system: Does your current arm use position-controlled joints, velocity-controlled joints, or torque-controlled joints? That determines how the actions should be represented and how difficult the initial dynamics-learning problem will be.