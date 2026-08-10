COMPOSE = docker compose
SERVICE = apnea_cnn

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

download: up 
	$(COMPOSE) exec $(SERVICE) python download.py

prep: up
	$(COMPOSE) exec $(SERVICE) python preprocess.py

train: up
	$(COMPOSE) exec $(SERVICE) python model.py

test: up
	$(COMPOSE) exec $(SERVICE) python test.py

all: build up prep train test

shell: up
	$(COMPOSE) exec $(SERVICE) bash

rebuild:
	$(COMPOSE) up -d --build

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

stop:
	$(COMPOSE) stop

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v
	docker system prune -af

.PHONY: build up download prep train test all shell rebuild logs ps stop down clean
