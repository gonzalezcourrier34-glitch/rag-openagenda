<h1 align="center">Assistant intelligent de recommandation d’événements culturels</h1>
<p align="center">
  <strong>Proof of Concept basé sur une architecture RAG</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-API-green" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangChain-RAG-orange" alt="LangChain">
  <img src="https://img.shields.io/badge/FAISS-Vector_Search-red" alt="FAISS">
  <img src="https://img.shields.io/badge/Mistral-LLM-purple" alt="Mistral">
  <img src="https://img.shields.io/badge/Streamlit-UI-pink" alt="Streamlit">
  <img src="https://img.shields.io/badge/PostgreSQL-Database-blue" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/MLflow-Tracking-lightgrey" alt="MLflow">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED" alt="Docker">
  <img src="https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF" alt="CI/CD">
</p>

<hr>

<h2>Présentation du projet</h2>

<p>
Dans ce projet, j’ai conçu un assistant intelligent capable de recommander des événements culturels à partir d’une question formulée en langage naturel.
</p>

<p>
L’objectif est de m’appuyer sur une architecture <strong>RAG (Retrieval-Augmented Generation)</strong> afin d’interroger des données réelles issues de l’API <strong>OpenAgenda</strong>, retrouver les événements les plus pertinents, puis générer une réponse claire, contextualisée et exploitable.
</p>

<p>
Ce projet m’a permis de travailler sur une chaîne complète de traitement, depuis la collecte des données jusqu’à leur exposition dans une application conteneurisée.
</p>

<ul>
  <li>collecte des données via API</li>
  <li>préparation et structuration de données textuelles</li>
  <li>vectorisation et recherche sémantique</li>
  <li>génération de réponse avec un LLM</li>
  <li>exposition du système via API FastAPI</li>
  <li>développement d’une interface utilisateur Streamlit</li>
  <li>containerisation avec Docker</li>
  <li>mise en place d’une logique CI/CD avec GitHub Actions</li>
</ul>

<hr>

<h2>1. Objectifs du projet</h2>

<h3>Contexte</h3>

<p>
Dans le cadre de ce Proof of Concept, l’entreprise fictive <strong>Puls-Events</strong> souhaite proposer un outil capable d’aider un utilisateur à trouver plus facilement des événements culturels correspondant à ses attentes.
</p>

<p>
Les moteurs de recherche classiques montrent rapidement leurs limites lorsque la demande est formulée librement, avec des besoins parfois flous, implicites ou insuffisamment structurés.
</p>

<h3>Problématique</h3>

<p>
La difficulté consiste donc à concevoir un système capable de comprendre une question exprimée naturellement, de retrouver les événements les plus pertinents, puis de formuler une réponse utile sans inventer d’informations.
</p>

<p>
C’est précisément l’intérêt d’une architecture RAG, qui combine une phase de recherche documentaire et une phase de génération.
</p>

<h3>Objectif du POC</h3>

<p>
Avec ce prototype, j’ai cherché à démontrer la faisabilité technique d’un assistant de recommandation culturelle reposant sur :
</p>

<ul>
  <li>des données réelles issues d’une API externe</li>
  <li>une recherche vectorielle sur des documents indexés</li>
  <li>un modèle de langage capable de générer une réponse contextualisée</li>
  <li>une architecture modulaire exposée via API et dashboard</li>
  <li>une base technique structurée pour évoluer vers un déploiement reproductible</li>
</ul>

<p>
L’objectif est qu’un utilisateur puisse poser une question comme :
</p>

<blockquote>
  <p><strong>« Je cherche une exposition d’architecture à Montpellier »</strong></p>
</blockquote>

<p>
et obtenir une réponse générée à partir d’événements réellement disponibles dans les données chargées.
</p>

<h3>Périmètre</h3>

<ul>
  <li><strong>Source de données :</strong> API OpenAgenda</li>
  <li><strong>Zone géographique :</strong> Montpellier par défaut, configurable</li>
  <li><strong>Domaine :</strong> événements culturels</li>
  <li><strong>Période :</strong> événements récupérés selon la fenêtre temporelle définie dans le projet</li>
</ul>

<hr>

<h2>2. Architecture du système</h2>

<p>
L’architecture repose sur une chaîne de traitement modulaire, dans laquelle chaque composant remplit un rôle précis.
</p>

<pre><code>Utilisateur
     │
     ▼
Streamlit
     │
     ▼
FastAPI
     │
     ▼
RAGService
 │
 ├── document_service → collecte, persistance et préparation des événements
 ├── FAISS → index vectoriel
 ├── memory_service → mémoire locale des échanges
 └── Mistral → génération de réponse
     │
     ▼
Réponse + documents sources
</code></pre>

<p>
Dans la version conteneurisée du projet, cette logique s’intègre dans une architecture plus large :
</p>

<pre><code>dashboard (Streamlit)
        │
        ▼
api (FastAPI)
        │
        ├── FAISS
        ├── mémoire locale
        ├── OpenAgenda
        ├── MLflow
        └── PostgreSQL
</code></pre>

<p>
Ce découpage m’a permis de séparer clairement :
</p>

<ul>
  <li>la récupération des données</li>
  <li>la logique RAG</li>
  <li>la mémoire conversationnelle</li>
  <li>l’exposition via API</li>
  <li>l’interface utilisateur</li>
  <li>le suivi expérimental</li>
  <li>la persistance technique</li>
  <li>l’exécution locale et conteneurisée</li>
</ul>

<hr>

<h2>3. Préparation et vectorisation des données</h2>

<h3>Source de données</h3>

<p>
Les données proviennent de l’API <strong>OpenAgenda</strong>. Les événements sont récupérés à l’aide de requêtes paginées, en filtrant selon une zone géographique et un périmètre définis dans la configuration.
</p>

<h3>Persistance des données</h3>

<p>
Afin de rendre le système plus robuste, j’ai ajouté une étape de persistance locale des données récupérées depuis l’API. Les événements peuvent être sauvegardés au format <strong>JSON</strong> et <strong>CSV</strong> dans le répertoire <code>data/</code>.
</p>

<p>
Cette étape me permet :
</p>

<ul>
  <li>de rejouer le pipeline sans dépendre systématiquement de l’API</li>
  <li>de conserver une trace des données collectées</li>
  <li>de faciliter la reconstruction de l’index vectoriel</li>
  <li>de structurer plus proprement le projet entre données brutes et données préparées</li>
</ul>

<h3>Préparation des données</h3>

<p>
Les événements bruts sont ensuite normalisés afin d’obtenir une structure cohérente. Cette étape m’a permis :
</p>

<ul>
  <li>d’uniformiser les noms de colonnes</li>
  <li>de gérer les valeurs manquantes</li>
  <li>de nettoyer les champs textuels</li>
  <li>de préparer les métadonnées utiles pour l’affichage et la recherche</li>
</ul>

<h3>Chunking</h3>

<p>
Dans ce projet, j’ai fait le choix de considérer <strong>chaque événement comme un document unique</strong>. Ce choix reste adapté au format des données et au périmètre du prototype.
</p>

<p>
Le texte utilisé pour la vectorisation est construit à partir des informations les plus utiles :
</p>

<ul>
  <li>titre</li>
  <li>description</li>
  <li>lieu</li>
  <li>ville</li>
  <li>région</li>
  <li>dates</li>
  <li>type d’événement</li>
</ul>

<h3>Embeddings</h3>

<p>
Pour la vectorisation, j’ai utilisé un modèle d’<strong>embeddings Mistral</strong>. Chaque document est transformé en vecteur puis ajouté dans un index <strong>FAISS</strong> afin de permettre une recherche sémantique rapide.
</p>

<hr>

<h2>4. Choix du modèle NLP</h2>

<p>
Pour la génération de réponses, j’ai choisi <strong>Mistral Small</strong>.
</p>

<p>
Ce choix m’a semblé pertinent pour plusieurs raisons :
</p>

<ul>
  <li>bonne qualité de génération</li>
  <li>coût raisonnable pour un prototype</li>
  <li>intégration simple avec LangChain</li>
</ul>

<p>
Le prompt utilisé impose une contrainte importante : le modèle doit répondre uniquement à partir du contexte documentaire retrouvé et ne pas inventer d’événements.
</p>

<p>
La principale limite reste la dépendance à la qualité des documents récupérés. Si le retrieval manque de pertinence, la génération finale en hérite directement.
</p>

<hr>

<h2>5. Construction de la base vectorielle</h2>

<p>
La base vectorielle est construite avec <strong>FAISS</strong>. Une fois les embeddings générés, les documents sont stockés dans l’index puis sauvegardés localement.
</p>

<p>
Cette persistance me permet :
</p>

<ul>
  <li>d’éviter de recalculer les embeddings à chaque redémarrage</li>
  <li>de reconstruire l’index uniquement lorsque cela est nécessaire</li>
  <li>de séparer plus proprement les données, l’index et les services applicatifs</li>
</ul>

<p>
Chaque document conserve également des métadonnées utiles, notamment :
</p>

<ul>
  <li>titre</li>
  <li>ville</li>
  <li>région</li>
  <li>lieu</li>
  <li>dates</li>
  <li>type d’événement</li>
  <li>URL source</li>
</ul>

<hr>

<h2>6. API et endpoints exposés</h2>

<p>
J’ai développé l’API avec <strong>FastAPI</strong> afin d’exposer le système sous forme de service REST.
</p>

<h3><code>/health</code></h3>

<p>
Permet de vérifier l’état du service et de savoir si l’index vectoriel est chargé.
</p>

<pre><code>{
  "status": "ok",
  "index_loaded": true
}</code></pre>

<h3><code>/ask</code></h3>

<p>
Permet de poser une question au système RAG.
</p>

<pre><code>{
  "question": "Je cherche une exposition à Montpellier"
}</code></pre>

<p>Exemple de structure de réponse :</p>

<pre><code>{
  "question": "...",
  "answer": "...",
  "n_docs": 3,
  "documents": [...]
}</code></pre>

<h3><code>/rebuild</code></h3>

<p>
Permet de reconstruire la base documentaire et l’index vectoriel à partir des données OpenAgenda.
</p>

<p>
Cet endpoint joue un rôle central lors du premier démarrage, car l’API doit d’abord charger les documents puis construire l’index FAISS avant de pouvoir répondre correctement aux requêtes utilisateurs.
</p>

<hr>

<h2>7. Qualité logicielle, tests et CI/CD</h2>

<h3>Tests</h3>

<p>
Afin de fiabiliser le projet, j’ai mis en place une base de tests automatisés sur les composants essentiels du système, notamment l’API, certains services applicatifs et les comportements attendus sur les endpoints principaux.
</p>

<p>
Cette démarche permet de vérifier plus facilement :
</p>

<ul>
  <li>le bon fonctionnement des endpoints critiques</li>
  <li>la stabilité du comportement attendu lors des évolutions du code</li>
  <li>la non-régression sur les fonctionnalités principales du prototype</li>
</ul>

<h3>CI</h3>

<p>
Le dépôt GitHub intègre une chaîne d’intégration continue avec <strong>GitHub Actions</strong>. À chaque mise à jour du code, le pipeline peut exécuter automatiquement les étapes de contrôle qualité, comme l’installation de l’environnement, l’exécution des tests et la vérification du bon état global du projet.
</p>

<h3>CD</h3>

<p>
Dans une logique de déploiement continu, l’architecture du projet a également été pensée pour pouvoir s’intégrer à une chaîne de livraison plus reproductible, notamment grâce à Docker, à la structuration du dépôt et à la séparation claire entre services.
</p>

<p>
Cette partie permet de rapprocher le prototype d’un fonctionnement plus industriel, même dans le cadre d’un POC.
</p>

<hr>

<h2>8. Évaluation du système</h2>

<p>
Pour évaluer le prototype, j’ai constitué un petit ensemble de questions couvrant différents besoins :
</p>

<ul>
  <li>recherche d’expositions</li>
  <li>activités familiales</li>
  <li>visites patrimoniales</li>
  <li>demandes culturelles plus générales</li>
</ul>

<p>
L’évaluation a été principalement qualitative. Je me suis concentré sur :
</p>

<ul>
  <li>la pertinence des documents récupérés</li>
  <li>la cohérence de la réponse générée</li>
  <li>la capacité du système à ne pas inventer d’informations</li>
</ul>

<hr>

<h2>9. Recommandations et perspectives</h2>

<h3>Ce qui fonctionne bien</h3>

<ul>
  <li>le système comprend des questions formulées naturellement</li>
  <li>la recherche sémantique retrouve des événements cohérents</li>
  <li>la réponse générée reste lisible et exploitable</li>
  <li>la mémoire locale améliore certains échanges de suivi</li>
  <li>l’architecture modulaire facilite les évolutions futures</li>
</ul>

<h3>Limites du prototype</h3>

<ul>
  <li>la qualité dépend fortement des données disponibles dans OpenAgenda</li>
  <li>le volume d’événements peut rester limité selon la zone ou la période choisie</li>
  <li>le filtrage métier reste encore simple</li>
  <li>les coûts peuvent augmenter avec les embeddings et les appels LLM</li>
</ul>

<h3>Améliorations possibles</h3>

<ul>
  <li>ajout d’un reranking des documents</li>
  <li>amélioration du filtrage par métadonnées</li>
  <li>mise en place d’une recherche hybride vectorielle et lexicale</li>
  <li>mémoire conversationnelle plus avancée</li>
  <li>déploiement cloud</li>
  <li>monitoring plus complet</li>
  <li>renforcement du pipeline CI/CD</li>
</ul>

<hr>

<h2>10. Organisation du dépôt GitHub</h2>

<pre><code>project/
│
├── app/
│   ├── main.py
│   ├── rag_service.py
│   ├── document_service.py
│   ├── memory_service.py
│   ├── schemas.py
│   ├── security.py
│   └── config.py
│
├── dashboard/
│   └── dashboard.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── index/
│
├── docs/
│
├── tests/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── pyproject.toml
├── README.md
└── mkdocs.yml
</code></pre>

<p>
Chaque module correspond à une brique du système, ce qui rend le projet plus lisible, plus modulaire et plus facile à faire évoluer.
</p>

<hr>

<h2>11. Interface utilisateur</h2>

<p>
J’ai également développé une interface <strong>Streamlit</strong> pour faciliter l’utilisation du système. Elle permet de :
</p>

<ul>
  <li>poser une question</li>
  <li>afficher la réponse générée dans un format plus lisible</li>
  <li>visualiser les documents utilisés</li>
  <li>consulter un historique local</li>
  <li>déclencher un rebuild de la base</li>
</ul>

<hr>

<h2>12. Suivi expérimental et infrastructure</h2>

<h3>MLflow</h3>

<p>
J’ai intégré <strong>MLflow</strong> afin de disposer d’un espace dédié au suivi expérimental et à la centralisation de certains essais liés au projet.
</p>

<h3>PostgreSQL</h3>

<p>
J’ai ajouté <strong>PostgreSQL</strong> comme base de données afin de préparer une architecture plus solide et plus proche d’un environnement applicatif réel.
</p>

<h3>Docker</h3>

<p>
L’ensemble de l’application est containerisé avec <strong>Docker</strong> et orchestré avec <strong>Docker Compose</strong>. Cette approche me permet de lancer la stack avec une seule commande et d’obtenir un environnement reproductible.
</p>

<ul>
  <li>séparation des services</li>
  <li>reproductibilité de l’environnement</li>
  <li>portabilité du projet</li>
  <li>intégration future dans une chaîne CI/CD complète</li>
</ul>

<hr>

<h2>Prérequis</h2>

<ul>
  <li>Docker</li>
  <li>Docker Compose</li>
  <li>Git</li>
  <li>une clé API OpenAgenda</li>
</ul>

<hr>

<h2>Installation</h2>

<h3>1. Cloner le projet</h3>

<pre><code>git clone repo_url
cd pocrag</code></pre>

<h3>2. Créer le fichier d’environnement</h3>

<p>
Copier le fichier d’exemple puis compléter les variables nécessaires :
</p>

<pre><code>cp .env.example .env</code></pre>

<p>Sous Windows PowerShell :</p>

<pre><code>Copy-Item .env.example .env</code></pre>

<p>Exemple de variables à renseigner :</p>

<pre><code>OPENAGENDA_API_KEY=votre_cle_openagenda
API_KEY=votre_token_api_perso

POSTGRES_DB=pocrag_db
POSTGRES_USER=pocrag_user
POSTGRES_PASSWORD=mot_de_passe_solide

DATABASE_URL=postgresql://pocrag_user:mot_de_passe_solide@db:5432/pocrag_db
MLFLOW_TRACKING_URI=http://mlflow:5000
API_URL=http://api:8000

ZONE_CHOISIE=Montpellier
TYPE_ZONE=city</code></pre>

<hr>

<h2>Lancer le projet</h2>

<h3>1. Construire et démarrer les services</h3>

<pre><code>docker compose up --build</code></pre>

<p>Pour lancer les services en arrière-plan :</p>

<pre><code>docker compose up --build -d</code></pre>

<h3>2. Vérifier les services</h3>

<ul>
  <li><strong>API FastAPI :</strong> <code>http://localhost:8000</code></li>
  <li><strong>Documentation Swagger :</strong> <code>http://localhost:8000/docs</code></li>
  <li><strong>Dashboard Streamlit :</strong> <code>http://localhost:8501</code></li>
  <li><strong>MLflow :</strong> <code>http://localhost:5000</code></li>
</ul>

<h3>3. Initialiser le système RAG</h3>

<p>
Lors du premier démarrage, l’index vectoriel n’est pas encore construit. Il est donc nécessaire d’exécuter une reconstruction avant d’utiliser l’endpoint <code>/ask</code>.
</p>

<p>
Cette initialisation peut être réalisée depuis l’interface Streamlit :
</p>

<ul>
  <li>ouvrir <code>http://localhost:8501</code></li>
  <li>aller dans l’onglet <strong>Administration</strong></li>
  <li>vérifier la zone géographique et le scope</li>
  <li>cliquer sur <strong>Rebuild /rebuild</strong></li>
</ul>

<p>
Elle peut également être réalisée directement via l’API :
</p>

<pre><code>curl -X POST "http://localhost:8000/rebuild" \
  -H "Content-Type: application/json" \
  -H "x-api-key: votre_token_api_perso" \
  -d "{\"zone\":\"Montpellier\",\"scope\":\"city\"}"</code></pre>

<h3>4. Interroger le système</h3>

<p>
Une fois l’index reconstruit, il est possible de poser une question depuis le dashboard ou directement via l’API.
</p>

<pre><code>curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -H "x-api-key: votre_token_api_perso" \
  -d "{\"question\":\"Je cherche une exposition d'architecture à Montpellier\"}"</code></pre>

<h3>5. Arrêter les services</h3>

<pre><code>docker compose down</code></pre>

<p>Pour supprimer également les volumes :</p>

<pre><code>docker compose down -v</code></pre>

<hr>

<h2>Résumé rapide</h2>

<pre><code>git clone repo_url
cd pocrag
cp .env.example .env
docker compose up --build</code></pre>

<p>
Puis :
</p>

<ol>
  <li>ouvrir <code>http://localhost:8501</code></li>
  <li>aller dans <strong>Administration</strong></li>
  <li>lancer <strong>Rebuild /rebuild</strong></li>
  <li>revenir dans <strong>Chat</strong></li>
  <li>poser une question</li>
</ol>

<hr>

<h2>Auteur</h2>

<p>
Projet réalisé par <strong>Stéphane GONZALEZ</strong> dans le cadre d’un apprentissage autour des architectures RAG, des applications LLM et des systèmes d’IA applicatifs.
</p>

<hr>

<h2>Licence</h2>

<p>Usage éducatif.</p>