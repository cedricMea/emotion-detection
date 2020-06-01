# Emotions detection app
This repo is an application for sentiment detection in short sentences. The app contains 3 differents container built with docker 


1. Differents containers
-	Frontend container
 	The frontend container is a javascript container. The node_modules has not been upload and must be rebuilt.
  	This container only define the vue for the app

- 	Flask api container
  	The api is full in python. It implements two models built with tensorflow on differents inputs data. Inputs data can be found in the directory flask-app/inputs.
	I did not upload glove/50-dimensions in the inputs folder
	These models are available through a flask api

-   Nginx container 
	This container performs routing in the docker network

2. How to run the app

- Install docker and docker compose on your machine

- Once Docker is well installed open a command prompt at the root of the project and type
   `docker-compose up --build`
   Docker will then build different container in a test version. (Developpement version will come soon :) )

 - go to your navigator and enter 
 `localhost:3000` 
 You will see there the the app running and will be able to test it  




