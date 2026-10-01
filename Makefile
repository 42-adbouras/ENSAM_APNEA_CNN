COMPOSE = docker compose

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

rebuild:
	$(COMPOSE) up -d --build

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

shell:
	$(COMPOSE) exec inference bash

stop:
	$(COMPOSE) stop

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v
	docker system prune -af

.PHONY: build up rebuild logs ps shell stop down clean
