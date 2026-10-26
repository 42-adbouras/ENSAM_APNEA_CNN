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

db-shell:
	$(COMPOSE) exec database sh -c 'psql -U "$$DB_USER" -d "$$DB_NAME"'

stop:
	$(COMPOSE) stop

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v
	docker system prune -af

.PHONY: build up rebuild logs ps shell db-shell stop down clean
