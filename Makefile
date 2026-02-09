start-dependencies:
	mkdir -p neo4j/logs neo4j/import
  chmod -R 777 neo4j/logs neo4j/import
	docker-compose up -d

stop-dependencies:
	docker-compose down
