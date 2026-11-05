COMPOSE = docker compose

build:
	$(COMPOSE) build

up:
	@mkdir -p data/apnea-ecg results
	$(COMPOSE) up -d

rebuild:
	@mkdir -p data/apnea-ecg results
	$(COMPOSE) up -d --build

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

shell:
	$(COMPOSE) exec inference bash

db-shell:
	$(COMPOSE) exec database sh -c 'psql -U "$$DB_USER" -d "$$DB_NAME"'

demo: up
	@$(COMPOSE) exec -T --user "$$(id -u):$$(id -g)" inference python -u /demo/apnea_ecg.py

eval: up
	@$(COMPOSE) exec -T --user "$$(id -u):$$(id -g)" inference python -u /demo/evaluate.py

stop:
	$(COMPOSE) stop

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v
	docker system prune -af

fclean: clean
	rm -rf services/database/data/*

.PHONY: build up rebuild logs ps shell db-shell demo eval stop down clean fclean
