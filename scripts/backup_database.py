#!/usr/bin/env python3
"""
Database backup utility for ArXiv recommendation system.

This script provides a CLI interface to the BackupService for creating
and managing database backups.
"""

import asyncio
import sys

from script_utils import (
    setup_backend_path, print_success, print_error, print_progress
)

# Setup backend imports
setup_backend_path()
from services import BackupService


async def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Database backup and restore utility"
    )
    
    subparsers = parser.add_subparsers(dest="action", help="Available actions")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create a backup")
    backup_parser.add_argument(
        "--type", 
        choices=["ratings", "papers", "full"], 
        default="ratings",
        help="Type of backup to create (default: ratings)"
    )
    
    # List command
    subparsers.add_parser("list", help="List available backups")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument(
        "--backup-file", 
        required=True,
        help="Backup file to restore from"
    )
    restore_parser.add_argument(
        "--type",
        choices=["ratings", "full"],
        default="ratings", 
        help="Type of restore (default: ratings)"
    )
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a backup file")
    delete_parser.add_argument(
        "--backup-file",
        required=True,
        help="Backup file to delete"
    )
    
    args = parser.parse_args()
    
    if not args.action:
        parser.print_help()
        return
    
    backup_service = BackupService()
    
    try:
        if args.action == "backup":
            print_progress(f"Creating {args.type} backup...")
            result = await backup_service.create_backup(args.type)
            
            if result.get("success"):
                print_success("Backup created successfully:")
                print(f"   File: {result['filename']}")
                print(f"   Path: {result['path']}")
                if "total_ratings" in result:
                    print(f"   Ratings: {result['total_ratings']}")
                elif "total_papers" in result:
                    print(f"   Papers: {result['total_papers']}")
                print(f"   Timestamp: {result['timestamp']}")
            else:
                print_error(f"Backup failed: {result.get('error', 'Unknown error')}")
                
        elif args.action == "list":
            print("📋 Available backups:")
            backups = backup_service.list_backups()
            
            if not backups:
                print("   No backups found")
            else:
                for backup in backups:
                    print(f"\n📄 {backup['filename']}")
                    print(f"   Type: {backup['backup_type']}")
                    print(f"   Size: {backup['size_mb']:.1f} MB")
                    print(f"   Timestamp: {backup['timestamp']}")
                    if backup['total_items'] != "N/A":
                        print(f"   Items: {backup['total_items']}")
                        
        elif args.action == "restore":
            print_progress(f"Restoring {args.type} from {args.backup_file}...")
            
            if args.type == "ratings":
                result = await backup_service.restore_ratings(args.backup_file)
            elif args.type == "full":
                result = await backup_service.restore_full_database(args.backup_file)
            else:
                print_error(f"Invalid restore type: {args.type}")
                return
                
            if result.get("success"):
                print_success("Restore completed successfully:")
                if "restored_count" in result:
                    print(f"   Restored: {result['restored_count']} ratings")
                    print(f"   Failed: {result['failed_count']} ratings")
                    print(f"   Total in backup: {result['total_in_backup']}")
                elif "restored_from" in result:
                    print(f"   Restored from: {result['restored_from']}")
                    print(f"   Safety backup: {result['safety_backup']}")
            else:
                print_error(f"Restore failed: {result.get('error', 'Unknown error')}")
                
        elif args.action == "delete":
            print_progress(f"Deleting backup {args.backup_file}...")
            result = backup_service.delete_backup(args.backup_file)
            
            if result.get("success"):
                print_success(f"Backup deleted: {result['deleted_file']}")
            else:
                print_error(f"Delete failed: {result.get('error', 'Unknown error')}")
                
    except Exception as e:
        print_error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())