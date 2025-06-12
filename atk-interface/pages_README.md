This codebase will contain an interface for interacting with a locally-running Augmentoolkit server. The structure of the Augmentoolkit server's API is described in api_README.md.

A basic sketch of pages is laid out below:
pages are probably
1. navigation. Configs, Inputs, and Outputs? And pipeline execution babysitting.
2. Config editing. Modify config fields directly, oh and if you point input_dir or output_dir at something that does not exist, it warns you.
    - this page will have the ability for executing a pipeline too. 
3. Input editing.  Upload files, view what files are there, delete files. Main thing is sorta like a filebrowser over the structure we get back as a result. Download button on the selected item, alternatively shift click or drag select and download all selected items,  everything will feel nice and snappy.
4. Output editing. Same operations as inputs plus it shows a list of all task IDs this session and lets you navigate to any output dir from there.
5. pipeline execution babysitting! We have this big progress bar at the top as well as an expandable details box off to the side showing (read only) the config we ran with with syntax highlighting. details key will include execution stats, that's what I'll use that reserved field for -- and the interface will proudly display whatever key-value pairs are added to the details as part of pipeline execution. It polls details and progress once a second when this screen is on (single page app you can click between screens and go back to the last one and it will be where you left off on it). A big download output dir button. A big cancel execution button with a confirm modal. And in a window, also updating (maybe once every .5 seconds or less) we have the logs streaming in (efficiently, using the cursor).

Some design brainstorming is below:
Tactile feedback and style:
1. dark mode maybe gradient for modern feel. Motion/replaying things in the background, maybe just moving shapes, maybe something else that exudes AI techological feel like light effects against a dark background, falling symbols or whatever, just give the ADD brains something to be mesmerized by during lulls. Idea: not falling symbols. There'll be one or mulitple beads of light travelling along the paths created by a diagram of a neural network. You know, the thing with all the nodes interconnected with lines? Would allow for a lot of variation, is topical, is hopefully easy to code? Hmm.)
2. buttons shake and move when pressed and generally satisfying
3. highlights, different color based on actions
4. stylize things like the drag select, download, modals, make you feel the actions
5. make errors less painful by cheerful presentation while displaying all important information.
6. Rounded corners etc.
7. When datagen runs finish, the mood is triumphant.
8. High numbers in the details box change its color and effects?
9. Particle effects?
10. text boxes smooth and elegant Whatever makes typing seem smoother or whatever, more fluid and flowing,  do that. Colored typing cursor?
11. and have a "boring mode" toggle in the 6. settings page that disables all the fun stuff.
12. actions, when taken, should highlight/draw attention to the natural next action through design. When a dataset generation run finishes, the "download output dir" (or however it is more concisely worded) button should react, be highlighted, appealing, etc. When files are selected in the input or output editing pages (such as potentially for downloading) the download button reacts. If a config is being edited, and the input_dir or output_dir is pointing to something which, we can tell from the structure of those does not exist, and the interface is warning you, the "run this config" should be less obvious or imply that it is a bad idea. This makes using the interface self-explanatory and easier and more fun and also makes doing so more engaging.

### Specific tools we will use to achieve visual impact: ###
Particle effects
coloration changes based on page and activity
Lighting following interaction
3d objects that move with the mouse
Shaking background for important actions
Background fades for transition from one page to another 
...more...

NOTE things like particle effects, which are cool but are unusual for this sort of thing, will get used only for the MOST important things (like progress bar) so that we don't jar/look pretentious and overdesigned/so that it remains tasteful

###

General actions:
modals autofocus the first text box and each step in a modal guides you to the next
new files that are created should be highlighted until some action is taken on the page
IF you can't tell the effect is there, the effect is either extraneous and should be removed, or should be strengthened
too many effects (or too many effects in different places) when an action is taken is confusing. Draw their attention to one place (see the or double click to open OVER THE DELETE BUTTON in configs management.


Everything should be very performant to keep the project fast to clone and set up. No large clunky videos in the background playing on loop -- generate things. Maybe we can use a simple web physics thing or lighting thing or whatever -- think like an svg over a png, we will make the most excessive style things be done in elegant ways, code- and space-wise.

Style thoughts:
We're going to make this nice to look at, engaging and stylistic, and make it give feedback on the basic actions in order to make the work more engaging, fun, and viral. We're going to take lessons from game mechanics and make the "second-to-second" experience really fun, in fact absurdly fun for software use. However this new approach must be done with style and panache. Quips in modals must be funny not forced. Effects must be sexy not garish and obnoxious. The fine line will be walked and we will not fall off.
But as we walk it we ascend.

The unconventional philosophy with regards to style here is taking a page from video game creation. Apparently great care goes into making the second-to-second experience of games great, addicting, appealing, enjoyable. For instance, in a card game when you mouse over a card, ig might get bigger; its background might animate; a sound might play; particle effects may surround it; and when you click it, the places where you may play it also light up, new sounds play etc. Every action suggests the next action through effects, and everything feels cool to do. Ideally this software, though iti s for LLM dataset generation, would also be fun/cool to use. So we are adopting that approach.


-- How this runs --

This will be run on a user's local machine and connect to a local Augmentoolkit server.
The tech stack should be optimized for smooth, stylish operation and **easy setup** without catastrophic dependency bloat.
Web or app based, either way is fine

Notably, any AI agent taking on development of this will lack a lot of context on what Augmentoolkit is and does. For most explanatory strings you will want to stub them or leave todos. We'll have to collaborate w.r.t. what information is important to show and how, and what the workflows are, though much of this is either revealed via the api structure or the description here.

Flexible design is key, and being able to change things without messing up the entire project is important. We want flexibility in UI updates, in state updates -- fragile programming is a big nono since this is being hacked together and hacked on. We want the AK47 of tech stack choice, design, and approach. It should work even if I drag it hrough the figurative mud. The API it is using is pretty simple after all.

### TECH STACK!!!

The local interface for the Augmentoolkit server will be a **web-based application** (specifically a Single Page Application, or SPA) built with the following technologies:

- **Type**: Web-based application (SPA)
- **Framework**: React with Vite
- **State Management**: Context API
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Config Editing**: Monaco Editor
- **File Management**: Custom components with `react arborist` for performance, customization, and modernity.
- **API Communication**: Fetch API
- **Real-Time Updates**: Polling with `setInterval`
- **Development Tools**: ESLint, Prettier, Jest, React Testing Library

#### Thought Process Behind Tech Stack Decisions

1. **Web-Based Application (SPA)**  
   - **Why**: A web-based interface simplifies deployment and user setup. Users can run the Augmentoolkit server and access the interface through a browser, eliminating the need for platform-specific builds or complex installations. This aligns with the goal of making setup easy and accessible.  
   - **Reasoning**: Leveraging HTML, CSS, and JavaScript ensures cross-platform compatibility and rapid development, as these are widely supported and familiar technologies.

2. **React with Vite**  
   - **Why**: React’s component-based architecture is perfect for building a modular SPA with distinct pages like Configs, Inputs, Outputs, and Pipeline Execution. Vite enhances this by offering faster builds and a modern development experience compared to alternatives like Create React App.  
   - **Reasoning**: React’s ecosystem and Vite’s performance make this combination ideal for quick iteration and scalability.

3. **Context API for State Management**  
   - **Why**: The Context API, built into React, provides a lightweight solution for managing state (e.g., task IDs, config data) across components. It avoids the complexity of heavier tools like Redux for this project’s scope.  
   - **Reasoning**: Simplicity and integration with React make it a practical choice for a focused interface.

4. **Tailwind CSS for Styling**  
   - **Why**: Tailwind CSS, a utility-first framework, replaces Styled Components in this stack. It’s elegant, concise, and familiar to you, offering rapid styling with extensive customization options. Unlike Styled Components, which you’re less comfortable with, Tailwind’s utility classes allow you to style directly in your markup, streamlining development.  
   - **Reasoning**: Tailwind supports quick iteration and consistency while enabling creative UI designs (e.g., gradients, dark mode, tactile feedback), aligning with your preference and the goal of an engaging, game-like interface.

5. **Framer Motion for Animations**  
   - **Why**: Framer Motion integrates seamlessly with React and simplifies the creation of sophisticated animations (e.g., moving beads of light, particle effects) to make the interface fun and dynamic.  
   - **Reasoning**: Its ease of use and power enhance the user experience without adding unnecessary complexity.

6. **Monaco Editor for Config Editing**  
   - **Why**: The Monaco Editor (the engine behind VS Code) offers a robust, feature-rich environment for editing YAML configs, complete with syntax highlighting and validation to catch errors like invalid paths.  
   - **Reasoning**: It’s a proven tool that elevates the config-editing experience for users.

7. **Custom Components with `react-folder-tree` for File Management**  
   - **Why**: This lightweight library enables a clear, interactive directory structure for the Inputs and Outputs pages, supporting actions like browsing, uploading, downloading, and deleting files.  
   - **Reasoning**: It’s simple yet effective for file-related functionality without overcomplicating the stack.

8. **Fetch API for API Communication**  
   - **Why**: The Fetch API, built into browsers, is straightforward and sufficient for REST API interactions, avoiding the need for external libraries like Axios.  
   - **Reasoning**: It’s lightweight and meets the project’s communication needs efficiently.

9. **Polling with `setInterval` for Real-Time Updates**  
   - **Why**: Polling with `setInterval` provides a simple way to fetch task status, progress, and logs in real time. Intervals can be adjusted dynamically based on task activity for optimal performance.  
   - **Reasoning**: It’s easy to implement and effective for this use case, balancing simplicity and responsiveness.

10. **Development Tools**  
    - **Why**: ESLint and Prettier enforce code quality and consistent formatting, while Jest and React Testing Library enable unit and integration testing for reliability.  
    - **Reasoning**: These tools ensure maintainability and robustness, supporting a smooth development process.

This tech stack strikes a balance between **performance**, **ease of development**, and **flexibility**. It leverages Tailwind CSS for elegant, efficient styling per your preference, while delivering a functional, engaging interface that’s quick to set up and maintain.

**Start the development server**:  
```bash
npm run dev
```


Note to self -- when you run at start the top tabs will react based on common flows. Usually you'll go to configs first, to run something. So, that one will be reacting. Pipeline monitor may be more demure than usually.


So beyond pipeline monitoring, we have a config revision to do (easyish), and also log management (easy)