#!/usr/bin/env python3
"""CLI for Codebase RAG."""
import argparse
import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Codebase RAG CLI")
    sub = parser.add_subparsers(dest='cmd')

    # index
    p = sub.add_parser('index', help='Index a repository')
    p.add_argument('url', help='GitHub URL')
    p.add_argument('-f', '--force', action='store_true', help='Force re-index')
    p.add_argument('--llm', default='claude', help='LLM provider')

    # list
    sub.add_parser('list', help='List indexed repositories')

    # ask
    p = sub.add_parser('ask', help='Ask a question')
    p.add_argument('question', help='Question to ask')
    p.add_argument('-r', '--repo', help='Repository name')
    p.add_argument('--llm', default='claude', help='LLM provider')
    p.add_argument('-n', '--chunks', type=int, default=15, help='Chunks to retrieve')

    # chat
    p = sub.add_parser('chat', help='Interactive chat')
    p.add_argument('-r', '--repo', help='Repository name')
    p.add_argument('--llm', default='claude', help='LLM provider')

    # delete
    p = sub.add_parser('delete', help='Delete repository')
    p.add_argument('repo', help='Repository name')

    # stats
    p = sub.add_parser('stats', help='Show stats')
    p.add_argument('repo', nargs='?', help='Repository name')

    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        return

    if args.cmd == 'list':
        from vectordb import VectorDB
        repos = VectorDB().list_repos()
        print("\n📚 Indexed Repositories:" if repos else "No repositories indexed")
        for r in repos:
            print(f"  • {r}")
        return

    if args.cmd == 'delete':
        from rag import RAGPipeline
        rag = RAGPipeline()
        rag.delete_repo(args.repo)
        print(f"✓ Deleted: {args.repo}")
        return

    if args.cmd == 'stats':
        from rag import RAGPipeline
        rag = RAGPipeline()
        if args.repo:
            rag.load(args.repo)
        stats = rag.get_stats()
        print(f"\n📊 Stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    if args.cmd == 'index':
        from rag import RAGPipeline
        rag = RAGPipeline(args.llm)
        rag.index(args.url, args.force)
        return

    # For ask/chat, need to load repo
    from rag import RAGPipeline
    rag = RAGPipeline(args.llm)

    if args.repo:
        if not rag.load(args.repo):
            sys.exit(1)
    else:
        repos = rag.list_repos()
        if not repos:
            print("❌ No repos indexed. Run 'index' first.")
            sys.exit(1)

        if args.cmd == 'chat':
            print("\n📚 Select repository:")
            for i, r in enumerate(repos, 1):
                print(f"  {i}. {r}")
            try:
                choice = int(input("\nNumber: ").strip())
                rag.load(repos[choice - 1])
            except (ValueError, IndexError):
                print("Invalid choice")
                sys.exit(1)
        else:
            rag.load(repos[0])

    if args.cmd == 'ask':
        print(f"\n🤔 Analyzing...")
        result = rag.ask(args.question, args.chunks)
        print(f"\n{result['answer']}")
        if result.get('diagram'):
            print(f"\n```mermaid\n{result['diagram']}\n```")
        return

    if args.cmd == 'chat':
        print(f"\n💬 Chat - {rag.current_repo}")
        print("Commands: 'quit' to exit, 'clear' to clear history\n")

        while True:
            try:
                q = input("You: ").strip()
                if not q:
                    continue
                if q.lower() in ['quit', 'exit', 'q']:
                    break
                if q.lower() == 'clear':
                    rag.clear_history()
                    print("✓ History cleared\n")
                    continue

                result = rag.ask(q)
                print(f"\nAssistant: {result['answer']}")
                if result.get('diagram'):
                    print(f"\n📊 Diagram generated")
                print()

            except KeyboardInterrupt:
                break

        print("\n👋 Bye!")


if __name__ == "__main__":
    main()
