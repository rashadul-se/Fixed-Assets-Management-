"""
Fixed Assets Management System with Virtual Coin Economy
Complete Enterprise Solution with Redis, SQLite, Load Balancing, and Disaster Recovery
Author: Enterprise Solutions Team
Version: 1.0.0
"""

import streamlit as st
import sqlite3
import redis
import hashlib
import json
import uuid
import threading
import queue
import time
import random
import pickle
import os
import shutil
import gzip
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============== CONFIGURATION ==============
class Config:
    DB_PATH = "fixed_assets.db"
    BACKUP_DIR = "backups"
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    MAX_CONNECTIONS = 100
    CACHE_TTL = 300
    MAX_WORKERS = 50
    BACKUP_INTERVAL = 3600
    INITIAL_COINS = 10000
    ENCRYPTION_KEY = "your-secret-key-here"

# ============== ENUMS ==============
class AssetStatus(Enum):
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    DISPOSED = "disposed"
    FOR_SALE = "for_sale"
    AUCTION = "auction"

class AuctionType(Enum):
    ENGLISH = "english"
    DUTCH = "dutch"
    SEALED_BID = "sealed_bid"

class TransactionType(Enum):
    PURCHASE = "purchase"
    SALE = "sale"
    AUCTION_WIN = "auction_win"
    TRANSFER = "transfer"
    ALLOCATION = "allocation"

class UserRole(Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    AUDITOR = "auditor"

# ============== DATA CLASSES ==============
@dataclass
class User:
    id: str
    username: str
    password_hash: str
    role: UserRole
    department: str
    wallet_balance: float
    created_at: datetime
    is_active: bool = True

@dataclass
class Asset:
    id: str
    name: str
    category: str
    description: str
    initial_cost: float
    current_value: float
    virtual_coin_price: float
    status: AssetStatus
    location: str
    department: str
    owner_id: str
    purchase_date: datetime
    depreciation_rate: float
    condition: str
    image_url: str = ""
    qr_code: str = ""

@dataclass
class Auction:
    id: str
    asset_id: str
    seller_id: str
    auction_type: AuctionType
    starting_price: float
    reserve_price: float
    current_bid: float
    highest_bidder_id: Optional[str]
    start_time: datetime
    end_time: datetime
    bid_increment: float
    is_active: bool = True

@dataclass
class Transaction:
    id: str
    type: TransactionType
    from_user_id: str
    to_user_id: str
    asset_id: Optional[str]
    amount: float
    timestamp: datetime
    description: str

# ============== SECURITY MODULE ==============
class SecurityManager:
    """Handles all security operations"""
    
    @staticmethod
    def hash_password(password: str, salt: str = None) -> Tuple[str, str]:
        if salt is None:
            salt = uuid.uuid4().hex
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return hashed.hex(), salt
    
    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        new_hash, _ = SecurityManager.hash_password(password, salt)
        return new_hash == hashed
    
    @staticmethod
    def encrypt_data(data: str) -> str:
        key = Config.ENCRYPTION_KEY.encode()
        encrypted = bytes([a ^ b for a, b in zip(data.encode(), key * (len(data) // len(key) + 1))])
        return encrypted.hex()
    
    @staticmethod
    def decrypt_data(encrypted: str) -> str:
        key = Config.ENCRYPTION_KEY.encode()
        data = bytes.fromhex(encrypted)
        decrypted = bytes([a ^ b for a, b in zip(data, key * (len(data) // len(key) + 1))])
        return decrypted.decode()
    
    @staticmethod
    def sanitize_input(input_str: str) -> str:
        dangerous = ["'", '"', ";", "--", "/*", "*/", "xp_", "DROP", "DELETE", "INSERT", "UPDATE"]
        result = input_str
        for char in dangerous:
            result = result.replace(char, "")
        return result

# ============== CONNECTION POOL ==============
class ConnectionPool:
    """Thread-safe database connection pool"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self._pool = queue.Queue(maxsize=Config.MAX_CONNECTIONS)
        self._size = 0
        self._pool_lock = threading.Lock()
        for _ in range(min(10, Config.MAX_CONNECTIONS)):
            self._create_connection()
    
    def _create_connection(self):
        conn = sqlite3.connect(Config.DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        self._pool.put(conn)
        self._size += 1
    
    @contextmanager
    def get_connection(self):
        conn = None
        try:
            try:
                conn = self._pool.get(timeout=5)
            except queue.Empty:
                with self._pool_lock:
                    if self._size < Config.MAX_CONNECTIONS:
                        self._create_connection()
                        conn = self._pool.get(timeout=5)
                    else:
                        conn = self._pool.get(timeout=30)
            yield conn
        finally:
            if conn:
                self._pool.put(conn)

# ============== REDIS CACHE MANAGER ==============
class CacheManager:
    """Redis cache management with fallback"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self._local_cache = {}
        self._redis = None
        try:
            self._redis = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                decode_responses=False,
                socket_connect_timeout=2
            )
            self._redis.ping()
            self._redis_available = True
        except:
            self._redis_available = False
    
    def get(self, key: str) -> Optional[Any]:
        try:
            if self._redis_available:
                data = self._redis.get(key)
                if data:
                    return pickle.loads(data)
            return self._local_cache.get(key, {}).get('data')
        except:
            return self._local_cache.get(key, {}).get('data')
    
    def set(self, key: str, value: Any, ttl: int = Config.CACHE_TTL):
        try:
            if self._redis_available:
                self._redis.setex(key, ttl, pickle.dumps(value))
            self._local_cache[key] = {'data': value, 'expires': time.time() + ttl}
        except:
            self._local_cache[key] = {'data': value, 'expires': time.time() + ttl}
    
    def delete(self, key: str):
        try:
            if self._redis_available:
                self._redis.delete(key)
            self._local_cache.pop(key, None)
        except:
            self._local_cache.pop(key, None)
    
    def invalidate_pattern(self, pattern: str):
        try:
            if self._redis_available:
                keys = self._redis.keys(pattern)
                if keys:
                    self._redis.delete(*keys)
            self._local_cache = {k: v for k, v in self._local_cache.items() if pattern.replace('*', '') not in k}
        except:
            pass

# ============== LOAD BALANCER ==============
class LoadBalancer:
    """Request load balancer with health monitoring"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self._executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)
        self._active_connections = 0
        self._lock = threading.Lock()
        self._metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0
        }
    
    def execute_task(self, func, *args, **kwargs):
        with self._lock:
            self._active_connections += 1
            self._metrics['total_requests'] += 1
        
        start_time = time.time()
        try:
            result = self._executor.submit(func, *args, **kwargs).result(timeout=30)
            with self._lock:
                self._metrics['successful_requests'] += 1
            return result
        except Exception as e:
            with self._lock:
                self._metrics['failed_requests'] += 1
            raise e
        finally:
            response_time = time.time() - start_time
            with self._lock:
                self._active_connections -= 1
                self._metrics['avg_response_time'] = (
                    (self._metrics['avg_response_time'] * (self._metrics['total_requests'] - 1) + response_time) / 
                    self._metrics['total_requests']
                )
    
    def get_health_status(self) -> Dict:
        return {
            'status': 'healthy' if self._active_connections < Config.MAX_WORKERS * 0.8 else 'degraded',
            'active_connections': self._active_connections,
            'max_connections': Config.MAX_WORKERS,
            'avg_response_time_ms': round(self._metrics['avg_response_time'] * 1000, 2),
            'success_rate': round(
                (self._metrics['successful_requests'] / max(1, self._metrics['total_requests'])) * 100, 2
            ),
            'cache_available': CacheManager()._redis_available
        }
    
    def get_metrics(self) -> Dict:
        return self._metrics.copy()

# ============== DISASTER RECOVERY ==============
class DisasterRecovery:
    """Backup and recovery management"""
    
    def __init__(self):
        self._backup_dir = Config.BACKUP_DIR
        os.makedirs(self._backup_dir, exist_ok=True)
        self._auto_backup_thread = None
        self._stop_backup = threading.Event()
    
    def create_backup(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}.db.gz"
        backup_path = os.path.join(self._backup_dir, backup_name)
        
        with open(Config.DB_PATH, 'rb') as f_in:
            with gzip.open(backup_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return backup_path
    
    def restore_backup(self, backup_path: str) -> bool:
        try:
            with gzip.open(backup_path, 'rb') as f_in:
                with open(Config.DB_PATH, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return True
        except Exception as e:
            print(f"Restore failed: {e}")
            return False
    
    def get_available_backups(self) -> List[Dict]:
        backups = []
        for filename in os.listdir(self._backup_dir):
            if filename.endswith('.db.gz'):
                path = os.path.join(self._backup_dir, filename)
                backups.append({
                    'name': filename,
                    'path': path,
                    'size': os.path.getsize(path),
                    'created': datetime.fromtimestamp(os.path.getctime(path))
                })
        return sorted(backups, key=lambda x: x['created'], reverse=True)
    
    def export_to_json(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.join(self._backup_dir, f"export_{timestamp}.json")
        
        data = {}
        with ConnectionPool().get_connection() as conn:
            for table in ['users', 'assets', 'auctions', 'transactions']:
                cursor = conn.execute(f"SELECT * FROM {table}")
                data[table] = [dict(row) for row in cursor.fetchall()]
        
        with open(export_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return export_path
    
    def start_auto_backup(self, interval: int = Config.BACKUP_INTERVAL):
        if self._auto_backup_thread and self._auto_backup_thread.is_alive():
            return
        
        def backup_loop():
            while not self._stop_backup.wait(interval):
                try:
                    self.create_backup()
                except Exception as e:
                    print(f"Auto backup failed: {e}")
        
        self._stop_backup.clear()
        self._auto_backup_thread = threading.Thread(target=backup_loop, daemon=True)
        self._auto_backup_thread.start()
    
    def stop_auto_backup(self):
        self._stop_backup.set()

# ============== DATABASE REPOSITORY (Abstract) ==============
class BaseRepository(ABC):
    """Abstract base repository with common operations"""
    
    def __init__(self):
        self._pool = ConnectionPool()
        self._cache = CacheManager()
    
    @abstractmethod
    def _get_table_name(self) -> str:
        pass
    
    def _execute_query(self, query: str, params: tuple = (), fetch: bool = True) -> Any:
        with self._pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            conn.commit()
            return cursor.lastrowid

# ============== USER REPOSITORY ==============
class UserRepository(BaseRepository):
    def _get_table_name(self) -> str:
        return "users"
    
    def create_user(self, user: User, salt: str) -> str:
        query = """
        INSERT INTO users (id, username, password_hash, salt, role, department, wallet_balance, created_at, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._execute_query(query, (
            user.id, user.username, user.password_hash, salt, user.role.value,
            user.department, user.wallet_balance, user.created_at.isoformat(), user.is_active
        ), fetch=False)
        self._cache.invalidate_pattern("user:*")
        return user.id
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        cache_key = f"user:username:{username}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        
        query = "SELECT * FROM users WHERE username = ? AND is_active = 1"
        result = self._execute_query(query, (username,))
        if result:
            user_dict = dict(result[0])
            self._cache.set(cache_key, user_dict)
            return user_dict
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        cache_key = f"user:id:{user_id}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        
        query = "SELECT * FROM users WHERE id = ?"
        result = self._execute_query(query, (user_id,))
        if result:
            user_dict = dict(result[0])
            self._cache.set(cache_key, user_dict)
            return user_dict
        return None
    
    def update_wallet(self, user_id: str, amount: float) -> bool:
        query = "UPDATE users SET wallet_balance = wallet_balance + ? WHERE id = ?"
        self._execute_query(query, (amount, user_id), fetch=False)
        self._cache.invalidate_pattern(f"user:*{user_id}*")
        return True
    
    def get_all_users(self) -> List[Dict]:
        query = "SELECT id, username, role, department, wallet_balance, created_at FROM users WHERE is_active = 1"
        return [dict(row) for row in self._execute_query(query)]

# ============== ASSET REPOSITORY ==============
class AssetRepository(BaseRepository):
    def _get_table_name(self) -> str:
        return "assets"
    
    def create_asset(self, asset: Asset) -> str:
        query = """
        INSERT INTO assets (id, name, category, description, initial_cost, current_value, 
        virtual_coin_price, status, location, department, owner_id, purchase_date, 
        depreciation_rate, condition, image_url, qr_code)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._execute_query(query, (
            asset.id, asset.name, asset.category, asset.description, asset.initial_cost,
            asset.current_value, asset.virtual_coin_price, asset.status.value, asset.location,
            asset.department, asset.owner_id, asset.purchase_date.isoformat(),
            asset.depreciation_rate, asset.condition, asset.image_url, asset.qr_code
        ), fetch=False)
        self._cache.invalidate_pattern("asset:*")
        return asset.id
    
    def get_asset_by_id(self, asset_id: str) -> Optional[Dict]:
        cache_key = f"asset:id:{asset_id}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        
        query = "SELECT * FROM assets WHERE id = ?"
        result = self._execute_query(query, (asset_id,))
        if result:
            asset_dict = dict(result[0])
            self._cache.set(cache_key, asset_dict)
            return asset_dict
        return None
    
    def get_assets_by_status(self, status: AssetStatus) -> List[Dict]:
        cache_key = f"assets:status:{status.value}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        
        query = "SELECT * FROM assets WHERE status = ? ORDER BY created_at DESC"
        result = [dict(row) for row in self._execute_query(query, (status.value,))]
        self._cache.set(cache_key, result, ttl=60)
        return result
    
    def get_assets_for_sale(self) -> List[Dict]:
        return self.get_assets_by_status(AssetStatus.FOR_SALE)
    
    def get_user_assets(self, user_id: str) -> List[Dict]:
        query = "SELECT * FROM assets WHERE owner_id = ? ORDER BY created_at DESC"
        return [dict(row) for row in self._execute_query(query, (user_id,))]
    
    def update_asset_status(self, asset_id: str, status: AssetStatus) -> bool:
        query = "UPDATE assets SET status = ? WHERE id = ?"
        self._execute_query(query, (status.value, asset_id), fetch=False)
        self._cache.invalidate_pattern("asset:*")
        return True
    
    def update_asset_owner(self, asset_id: str, new_owner_id: str) -> bool:
        query = "UPDATE assets SET owner_id = ?, status = ? WHERE id = ?"
        self._execute_query(query, (new_owner_id, AssetStatus.ACTIVE.value, asset_id), fetch=False)
        self._cache.invalidate_pattern("asset:*")
        return True
    
    def get_all_assets(self) -> List[Dict]:
        query = "SELECT * FROM assets ORDER BY created_at DESC"
        return [dict(row) for row in self._execute_query(query)]
    
    def search_assets(self, keyword: str, category: str = None, status: str = None) -> List[Dict]:
        query = "SELECT * FROM assets WHERE (name LIKE ? OR description LIKE ?)"
        params = [f"%{keyword}%", f"%{keyword}%"]
        if category:
            query += " AND category = ?"
            params.append(category)
        if status:
            query += " AND status = ?"
            params.append(status)
        return [dict(row) for row in self._execute_query(query, tuple(params))]

# ============== AUCTION REPOSITORY ==============
class AuctionRepository(BaseRepository):
    def _get_table_name(self) -> str:
        return "auctions"
    
    def create_auction(self, auction: Auction) -> str:
        query = """
        INSERT INTO auctions (id, asset_id, seller_id, auction_type, starting_price, 
        reserve_price, current_bid, highest_bidder_id, start_time, end_time, bid_increment, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._execute_query(query, (
            auction.id, auction.asset_id, auction.seller_id, auction.auction_type.value,
            auction.starting_price, auction.reserve_price, auction.current_bid,
            auction.highest_bidder_id, auction.start_time.isoformat(),
            auction.end_time.isoformat(), auction.bid_increment, auction.is_active
        ), fetch=False)
        self._cache.invalidate_pattern("auction:*")
        return auction.id
    
    def get_active_auctions(self) -> List[Dict]:
        cache_key = "auctions:active"
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        
        query = """
        SELECT a.*, ast.name as asset_name, ast.category, ast.condition, ast.image_url
        FROM auctions a
        JOIN assets ast ON a.asset_id = ast.id
        WHERE a.is_active = 1 AND a.end_time > ?
        ORDER BY a.end_time ASC
        """
        result = [dict(row) for row in self._execute_query(query, (datetime.now().isoformat(),))]
        self._cache.set(cache_key, result, ttl=30)
        return result
    
    def get_auction_by_id(self, auction_id: str) -> Optional[Dict]:
        query = """
        SELECT a.*, ast.name as asset_name, ast.category, ast.condition
        FROM auctions a
        JOIN assets ast ON a.asset_id = ast.id
        WHERE a.id = ?
        """
        result = self._execute_query(query, (auction_id,))
        return dict(result[0]) if result else None
    
    def place_bid(self, auction_id: str, bidder_id: str, bid_amount: float) -> bool:
        query = """
        UPDATE auctions SET current_bid = ?, highest_bidder_id = ?
        WHERE id = ? AND current_bid < ? AND is_active = 1
        """
        self._execute_query(query, (bid_amount, bidder_id, auction_id, bid_amount), fetch=False)
        self._cache.invalidate_pattern("auction:*")
        return True
    
    def close_auction(self, auction_id: str) -> bool:
        query = "UPDATE auctions SET is_active = 0 WHERE id = ?"
        self._execute_query(query, (auction_id,), fetch=False)
        self._cache.invalidate_pattern("auction:*")
        return True

# ============== TRANSACTION REPOSITORY ==============
class TransactionRepository(BaseRepository):
    def _get_table_name(self) -> str:
        return "transactions"
    
    def create_transaction(self, transaction: Transaction) -> str:
        query = """
        INSERT INTO transactions (id, type, from_user_id, to_user_id, asset_id, amount, timestamp, description)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._execute_query(query, (
            transaction.id, transaction.type.value, transaction.from_user_id,
            transaction.to_user_id, transaction.asset_id, transaction.amount,
            transaction.timestamp.isoformat(), transaction.description
        ), fetch=False)
        return transaction.id
    
    def get_user_transactions(self, user_id: str) -> List[Dict]:
        query = """
        SELECT * FROM transactions 
        WHERE from_user_id = ? OR to_user_id = ?
        ORDER BY timestamp DESC LIMIT 100
        """
        return [dict(row) for row in self._execute_query(query, (user_id, user_id))]
    
    def get_all_transactions(self, limit: int = 1000) -> List[Dict]:
        query = f"SELECT * FROM transactions ORDER BY timestamp DESC LIMIT {limit}"
        return [dict(row) for row in self._execute_query(query)]

# ============== BUSINESS SERVICES ==============
class AuthService:
    """Authentication business logic"""
    
    def __init__(self):
        self._user_repo = UserRepository()
    
    def register(self, username: str, password: str, role: UserRole, department: str) -> Tuple[bool, str]:
        username = SecurityManager.sanitize_input(username)
        existing = self._user_repo.get_user_by_username(username)
        if existing:
            return False, "Username already exists"
        
        password_hash, salt = SecurityManager.hash_password(password)
        user = User(
            id=str(uuid.uuid4()),
            username=username,
            password_hash=password_hash,
            role=role,
            department=department,
            wallet_balance=Config.INITIAL_COINS,
            created_at=datetime.now(),
            is_active=True
        )
        self._user_repo.create_user(user, salt)
        return True, user.id
    
    def login(self, username: str, password: str) -> Tuple[bool, Optional[Dict]]:
        username = SecurityManager.sanitize_input(username)
        user = self._user_repo.get_user_by_username(username)
        if not user:
            return False, None
        
        if SecurityManager.verify_password(password, user['password_hash'], user['salt']):
            return True, user
        return False, None

class AssetService:
    """Asset management business logic"""
    
    def __init__(self):
        self._asset_repo = AssetRepository()
        self._user_repo = UserRepository()
        self._transaction_repo = TransactionRepository()
    
    def create_asset(self, name: str, category: str, description: str, initial_cost: float,
                    location: str, department: str, owner_id: str, depreciation_rate: float,
                    condition: str) -> str:
        asset = Asset(
            id=str(uuid.uuid4()),
            name=SecurityManager.sanitize_input(name),
            category=category,
            description=SecurityManager.sanitize_input(description),
            initial_cost=initial_cost,
            current_value=initial_cost,
            virtual_coin_price=initial_cost,
            status=AssetStatus.ACTIVE,
            location=SecurityManager.sanitize_input(location),
            department=department,
            owner_id=owner_id,
            purchase_date=datetime.now(),
            depreciation_rate=depreciation_rate,
            condition=condition,
            qr_code=str(uuid.uuid4())[:8].upper()
        )
        return self._asset_repo.create_asset(asset)
    
    def list_for_sale(self, asset_id: str, price: float, seller_id: str) -> bool:
        asset = self._asset_repo.get_asset_by_id(asset_id)
        if not asset or asset['owner_id'] != seller_id:
            return False
        
        with ConnectionPool().get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE assets SET status = ?, virtual_coin_price = ? WHERE id = ?",
                (AssetStatus.FOR_SALE.value, price, asset_id)
            )
            conn.commit()
        CacheManager().invalidate_pattern("asset:*")
        return True
    
    def purchase_asset(self, asset_id: str, buyer_id: str) -> Tuple[bool, str]:
        asset = self._asset_repo.get_asset_by_id(asset_id)
        if not asset:
            return False, "Asset not found"
        if asset['status'] != AssetStatus.FOR_SALE.value:
            return False, "Asset not for sale"
        
        buyer = self._user_repo.get_user_by_id(buyer_id)
        if buyer['wallet_balance'] < asset['virtual_coin_price']:
            return False, "Insufficient balance"
        
        seller_id = asset['owner_id']
        price = asset['virtual_coin_price']
        
        self._user_repo.update_wallet(buyer_id, -price)
        self._user_repo.update_wallet(seller_id, price)
        self._asset_repo.update_asset_owner(asset_id, buyer_id)
        
        transaction = Transaction(
            id=str(uuid.uuid4()),
            type=TransactionType.PURCHASE,
            from_user_id=buyer_id,
            to_user_id=seller_id,
            asset_id=asset_id,
            amount=price,
            timestamp=datetime.now(),
            description=f"Purchase of {asset['name']}"
        )
        self._transaction_repo.create_transaction(transaction)
        
        return True, "Purchase successful"

class AuctionService:
    """Auction management business logic"""
    
    def __init__(self):
        self._auction_repo = AuctionRepository()
        self._asset_repo = AssetRepository()
        self._user_repo = UserRepository()
        self._transaction_repo = TransactionRepository()
    
    def create_auction(self, asset_id: str, seller_id: str, auction_type: AuctionType,
                      starting_price: float, reserve_price: float, duration_hours: int,
                      bid_increment: float) -> Tuple[bool, str]:
        asset = self._asset_repo.get_asset_by_id(asset_id)
        if not asset or asset['owner_id'] != seller_id:
            return False, "Invalid asset or not owner"
        
        self._asset_repo.update_asset_status(asset_id, AssetStatus.AUCTION)
        
        auction = Auction(
            id=str(uuid.uuid4()),
            asset_id=asset_id,
            seller_id=seller_id,
            auction_type=auction_type,
            starting_price=starting_price,
            reserve_price=reserve_price,
            current_bid=starting_price,
            highest_bidder_id=None,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=duration_hours),
            bid_increment=bid_increment,
            is_active=True
        )
        auction_id = self._auction_repo.create_auction(auction)
        return True, auction_id
    
    def place_bid(self, auction_id: str, bidder_id: str, bid_amount: float) -> Tuple[bool, str]:
        auction = self._auction_repo.get_auction_by_id(auction_id)
        if not auction:
            return False, "Auction not found"
        if not auction['is_active']:
            return False, "Auction has ended"
        if datetime.fromisoformat(auction['end_time']) < datetime.now():
            return False, "Auction has expired"
        
        min_bid = auction['current_bid'] + auction['bid_increment']
        if bid_amount < min_bid:
            return False, f"Minimum bid is {min_bid}"
        
        bidder = self._user_repo.get_user_by_id(bidder_id)
        if bidder['wallet_balance'] < bid_amount:
            return False, "Insufficient balance"
        
        self._auction_repo.place_bid(auction_id, bidder_id, bid_amount)
        return True, "Bid placed successfully"
    
    def finalize_auction(self, auction_id: str) -> Tuple[bool, str]:
        auction = self._auction_repo.get_auction_by_id(auction_id)
        if not auction:
            return False, "Auction not found"
        
        self._auction_repo.close_auction(auction_id)
        
        if auction['highest_bidder_id'] and auction['current_bid'] >= auction['reserve_price']:
            winner_id = auction['highest_bidder_id']
            final_price = auction['current_bid']
            seller_id = auction['seller_id']
            asset_id = auction['asset_id']
            
            self._user_repo.update_wallet(winner_id, -final_price)
            self._user_repo.update_wallet(seller_id, final_price)
            self._asset_repo.update_asset_owner(asset_id, winner_id)
            
            transaction = Transaction(
                id=str(uuid.uuid4()),
                type=TransactionType.AUCTION_WIN,
                from_user_id=winner_id,
                to_user_id=seller_id,
                asset_id=asset_id,
                amount=final_price,
                timestamp=datetime.now(),
                description=f"Auction win for {auction['asset_name']}"
            )
            self._transaction_repo.create_transaction(transaction)
            return True, "Auction finalized with winner"
        else:
            self._asset_repo.update_asset_status(auction['asset_id'], AssetStatus.ACTIVE)
            return True, "Auction ended without meeting reserve"

# ============== ANALYTICS SERVICE ==============
class AnalyticsService:
    """Business Intelligence Analytics"""
    
    def __init__(self):
        self._pool = ConnectionPool()
    
    def get_asset_distribution(self) -> pd.DataFrame:
        query = """
        SELECT category, COUNT(*) as count, SUM(current_value) as total_value,
               AVG(current_value) as avg_value
        FROM assets GROUP BY category
        """
        with self._pool.get_connection() as conn:
            return pd.read_sql_query(query, conn)
    
    def get_transaction_trends(self, days: int = 30) -> pd.DataFrame:
        query = f"""
        SELECT DATE(timestamp) as date, type, COUNT(*) as count, SUM(amount) as total_amount
        FROM transactions
        WHERE timestamp >= date('now', '-{days} days')
        GROUP BY DATE(timestamp), type
        ORDER BY date
        """
        with self._pool.get_connection() as conn:
            return pd.read_sql_query(query, conn)
    
    def get_department_spending(self) -> pd.DataFrame:
        query = """
        SELECT u.department, SUM(t.amount) as total_spent, COUNT(t.id) as transaction_count
        FROM transactions t
        JOIN users u ON t.from_user_id = u.id
        WHERE t.type IN ('purchase', 'auction_win')
        GROUP BY u.department
        """
        with self._pool.get_connection() as conn:
            return pd.read_sql_query(query, conn)
    
    def get_auction_performance(self) -> pd.DataFrame:
        query = """
        SELECT auction_type, COUNT(*) as total_auctions,
               AVG(current_bid - starting_price) as avg_price_increase,
               SUM(CASE WHEN highest_bidder_id IS NOT NULL THEN 1 ELSE 0 END) as successful_auctions
        FROM auctions
        GROUP BY auction_type
        """
        with self._pool.get_connection() as conn:
            return pd.read_sql_query(query, conn)
    
    def get_top_traders(self, limit: int = 10) -> pd.DataFrame:
        query = f"""
        SELECT u.username, u.department,
               COUNT(CASE WHEN t.from_user_id = u.id THEN 1 END) as purchases,
               COUNT(CASE WHEN t.to_user_id = u.id THEN 1 END) as sales,
               SUM(CASE WHEN t.to_user_id = u.id THEN t.amount ELSE 0 END) as earned,
               SUM(CASE WHEN t.from_user_id = u.id THEN t.amount ELSE 0 END) as spent
        FROM users u
        LEFT JOIN transactions t ON u.id = t.from_user_id OR u.id = t.to_user_id
        GROUP BY u.id
        ORDER BY (earned + spent) DESC
        LIMIT {limit}
        """
        with self._pool.get_connection() as conn:
            return pd.read_sql_query(query, conn)
    
    def get_asset_depreciation_report(self) -> pd.DataFrame:
        query = """
        SELECT name, category, initial_cost, current_value,
               (initial_cost - current_value) as depreciation_amount,
               ROUND((initial_cost - current_value) * 100.0 / initial_cost, 2) as depreciation_pct,
               depreciation_rate, purchase_date
        FROM assets
        WHERE initial_cost > 0
        ORDER BY depreciation_pct DESC
        """
        with self._pool.get_connection() as conn:
            return pd.read_sql_query(query, conn)
    
    def get_coin_circulation(self) -> Dict:
        with self._pool.get_connection() as conn:
            total_coins = pd.read_sql_query("SELECT SUM(wallet_balance) as total FROM users", conn)['total'].iloc[0]
            active_users = pd.read_sql_query("SELECT COUNT(*) as count FROM users WHERE is_active = 1", conn)['count'].iloc[0]
            total_transactions = pd.read_sql_query("SELECT COUNT(*) as count, SUM(amount) as volume FROM transactions", conn)
        
        return {
            'total_coins_in_circulation': total_coins or 0,
            'active_users': active_users or 0,
            'total_transactions': total_transactions['count'].iloc[0] or 0,
            'transaction_volume': total_transactions['volume'].iloc[0] or 0
        }

# ============== DATABASE INITIALIZATION ==============
def initialize_database():
    """Initialize database schema"""
    
    conn = sqlite3.connect(Config.DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        role TEXT NOT NULL,
        department TEXT NOT NULL,
        wallet_balance REAL DEFAULT 0,
        created_at TEXT NOT NULL,
        is_active INTEGER DEFAULT 1
    )
    """)
    
    # Assets table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS assets (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        description TEXT,
        initial_cost REAL NOT NULL,
        current_value REAL NOT NULL,
        virtual_coin_price REAL NOT NULL,
        status TEXT NOT NULL,
        location TEXT,
        department TEXT,
        owner_id TEXT NOT NULL,
        purchase_date TEXT NOT NULL,
        depreciation_rate REAL DEFAULT 0,
        condition TEXT,
        image_url TEXT,
        qr_code TEXT UNIQUE,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (owner_id) REFERENCES users(id)
    )
    """)
    
    # Auctions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS auctions (
        id TEXT PRIMARY KEY,
        asset_id TEXT NOT NULL,
        seller_id TEXT NOT NULL,
        auction_type TEXT NOT NULL,
        starting_price REAL NOT NULL,
        reserve_price REAL NOT NULL,
        current_bid REAL NOT NULL,
        highest_bidder_id TEXT,
        start_time TEXT NOT NULL,
        end_time TEXT NOT NULL,
        bid_increment REAL NOT NULL,
        is_active INTEGER DEFAULT 1,
        FOREIGN KEY (asset_id) REFERENCES assets(id),
        FOREIGN KEY (seller_id) REFERENCES users(id),
        FOREIGN KEY (highest_bidder_id) REFERENCES users(id)
    )
    """)
    
    # Transactions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        from_user_id TEXT NOT NULL,
        to_user_id TEXT NOT NULL,
        asset_id TEXT,
        amount REAL NOT NULL,
        timestamp TEXT NOT NULL,
        description TEXT,
        FOREIGN KEY (from_user_id) REFERENCES users(id),
        FOREIGN KEY (to_user_id) REFERENCES users(id),
        FOREIGN KEY (asset_id) REFERENCES assets(id)
    )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_assets_status ON assets(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_assets_owner ON assets(owner_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_auctions_active ON auctions(is_active, end_time)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_users ON transactions(from_user_id, to_user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp)")
    
    conn.commit()
    conn.close()

# ============== STREAMLIT UI COMPONENTS ==============
def setup_page_config():
    st.set_page_config(
        page_title="Fixed Assets Management System",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E40AF;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-card {
        background: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

def render_login_page():
    st.markdown('<h1 class="main-header">🏢 Fixed Assets Management System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login Portal")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("🚀 Login", use_container_width=True)
                
                if submit:
                    auth_service = AuthService()
                    success, user = auth_service.login(username, password)
                    if success:
                        st.session_state['logged_in'] = True
                        st.session_state['user'] = user
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                role = st.selectbox("Role", ["user", "manager", "admin", "auditor"])
                department = st.selectbox("Department", ["IT", "Finance", "Operations", "HR", "Sales"])
                
                register = st.form_submit_button("📝 Register", use_container_width=True)
                
                if register:
                    if new_password != confirm_password:
                        st.error("Passwords don't match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        auth_service = AuthService()
                        success, msg = auth_service.register(
                            new_username, new_password, UserRole(role), department
                        )
                        if success:
                            st.success("Registration successful! Please login.")
                        else:
                            st.error(msg)

def render_sidebar():
    with st.sidebar:
        user = st.session_state.get('user', {})
        st.markdown(f"### 👤 {user.get('username', 'User')}")
        st.markdown(f"**Role:** {user.get('role', 'N/A').title()}")
        st.markdown(f"**Department:** {user.get('department', 'N/A')}")
        st.markdown(f"**💰 Balance:** {user.get('wallet_balance', 0):,.2f} VC")
        
        st.markdown("---")
        
        menu_items = ["📊 Dashboard", "📦 My Assets", "🛒 Marketplace", 
                     "🎯 Auctions", "💱 Transactions", "📈 Analytics"]
        
        if user.get('role') in ['admin', 'manager']:
            menu_items.append("⚙️ Admin Panel")
        
        selected = st.radio("Navigation", menu_items, label_visibility="collapsed")
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['user'] = {}
            st.rerun()
        
        return selected

def render_dashboard():
    st.markdown('<h2>📊 Dashboard</h2>', unsafe_allow_html=True)
    
    user = st.session_state.get('user', {})
    asset_repo = AssetRepository()
    transaction_repo = TransactionRepository()
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        my_assets = asset_repo.get_user_assets(user.get('id'))
        st.metric("My Assets", len(my_assets))
    
    with col2:
        total_value = sum(a['current_value'] for a in my_assets)
        st.metric("Total Asset Value", f"{total_value:,.0f} VC")
    
    with col3:
        transactions = transaction_repo.get_user_transactions(user.get('id'))
        st.metric("Transactions", len(transactions))
    
    with col4:
        st.metric("Wallet Balance", f"{user.get('wallet_balance', 0):,.0f} VC")
    
    st.markdown("---")
    
    # Recent Activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📦 My Assets")
        if my_assets:
            for asset in my_assets[:5]:
                st.markdown(f"""
                <div class="info-card">
                    <strong>{asset['name']}</strong> - {asset['category']}<br>
                    <small>Status: {asset['status']} | Value: {asset['current_value']:,.0f} VC</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No assets yet")
    
    with col2:
        st.markdown("### 💱 Recent Transactions")
        if transactions:
            for tx in transactions[:5]:
                is_outgoing = tx['from_user_id'] == user.get('id')
                icon = "🔴" if is_outgoing else "🟢"
                st.markdown(f"""
                <div class="info-card">
                    {icon} <strong>{tx['type'].upper()}</strong><br>
                    <small>{tx['description']} - {'-' if is_outgoing else '+'}{tx['amount']:,.0f} VC</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No transactions yet")

def render_my_assets():
    st.markdown('<h2>📦 My Assets</h2>', unsafe_allow_html=True)
    
    user = st.session_state.get('user', {})
    asset_repo = AssetRepository()
    asset_service = AssetService()
    
    tab1, tab2 = st.tabs(["📋 Asset List", "➕ Add New Asset"])
    
    with tab1:
        my_assets = asset_repo.get_user_assets(user.get('id'))
        
        if my_assets:
            for asset in my_assets:
                with st.expander(f"📦 {asset['name']} - {asset['current_value']:,.0f} VC", expanded=False):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown(f"**Category:** {asset['category']}")
                        st.markdown(f"**Description:** {asset['description']}")
                        st.markdown(f"**Location:** {asset['location']}")
                        st.markdown(f"**QR Code:** {asset['qr_code']}")
                    
                    with col2:
                        st.markdown(f"**Status:** {asset['status']}")
                        st.markdown(f"**Condition:** {asset['condition']}")
                        st.markdown(f"**Initial Cost:** {asset['initial_cost']:,.0f} VC")
                        st.markdown(f"**Current Value:** {asset['current_value']:,.0f} VC")
                    
                    with col3:
                        if asset['status'] == 'active':
                            sale_price = st.number_input(
                                "Sale Price (VC)", 
                                min_value=1.0,
                                value=float(asset['current_value']),
                                key=f"price_{asset['id']}"
                            )
                            if st.button("🏷️ List for Sale", key=f"sell_{asset['id']}", use_container_width=True):
                                if asset_service.list_for_sale(asset['id'], sale_price, user.get('id')):
                                    st.success("Listed for sale!")
                                    st.rerun()
                                else:
                                    st.error("Failed to list")
        else:
            st.info("You don't have any assets yet")
    
    with tab2:
        st.markdown("### Create New Asset")
        
        with st.form("add_asset_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Asset Name*")
                category = st.selectbox("Category*", [
                    "Electronics", "Furniture", "Vehicles", "Equipment", 
                    "Real Estate", "Software", "Other"
                ])
                initial_cost = st.number_input("Initial Cost (VC)*", min_value=1.0, value=1000.0)
                location = st.text_input("Location*")
            
            with col2:
                description = st.text_area("Description")
                condition = st.selectbox("Condition", ["New", "Excellent", "Good", "Fair", "Poor"])
                depreciation_rate = st.slider("Annual Depreciation Rate (%)", 0.0, 50.0, 10.0)
                department = st.selectbox("Department", ["IT", "Finance", "Operations", "HR", "Sales"])
            
            submit = st.form_submit_button("✅ Create Asset", use_container_width=True)
            
            if submit:
                if name and category and location:
                    asset_id = asset_service.create_asset(
                        name=name,
                        category=category,
                        description=description,
                        initial_cost=initial_cost,
                        location=location,
                        department=department,
                        owner_id=user.get('id'),
                        depreciation_rate=depreciation_rate / 100,
                        condition=condition
                    )
                    st.success(f"Asset created! ID: {asset_id[:8]}...")
                    st.rerun()
                else:
                    st.error("Please fill all required fields")

def render_marketplace():
    st.markdown('<h2>🛒 Marketplace</h2>', unsafe_allow_html=True)
    
    user = st.session_state.get('user', {})
    asset_repo = AssetRepository()
    asset_service = AssetService()
    
    # Search and Filters
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        search = st.text_input("🔍 Search assets", placeholder="Enter keywords...")
    with col2:
        category_filter = st.selectbox("Category", ["All"] + [
            "Electronics", "Furniture", "Vehicles", "Equipment", 
            "Real Estate", "Software", "Other"
        ])
    with col3:
        sort_by = st.selectbox("Sort by", ["Price: Low to High", "Price: High to Low", "Newest First"])
    
    st.markdown("---")
    
    # Get assets for sale
    if search:
        assets = asset_repo.search_assets(
            search,
            category=None if category_filter == "All" else category_filter,
            status="for_sale"
        )
    else:
        assets = asset_repo.get_assets_for_sale()
        if category_filter != "All":
            assets = [a for a in assets if a['category'] == category_filter]
    
    # Sort assets
    if sort_by == "Price: Low to High":
        assets = sorted(assets, key=lambda x: x['virtual_coin_price'])
    elif sort_by == "Price: High to Low":
        assets = sorted(assets, key=lambda x: x['virtual_coin_price'], reverse=True)
    else:
        assets = sorted(assets, key=lambda x: x['purchase_date'], reverse=True)
    
    # Display assets
    if assets:
        for i in range(0, len(assets), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(assets):
                    asset = assets[i + j]
                    with col:
                        st.markdown(f"""
                        <div class="info-card">
                            <h4>{asset['name']}</h4>
                            <p><strong>Category:</strong> {asset['category']}</p>
                            <p><strong>Condition:</strong> {asset['condition']}</p>
                            <p><strong>Price:</strong> {asset['virtual_coin_price']:,.0f} VC</p>
                            <p><small>{asset['description'][:100]}...</small></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if asset['owner_id'] != user.get('id'):
                            if st.button(f"💳 Buy Now", key=f"buy_{asset['id']}", use_container_width=True):
                                success, msg = asset_service.purchase_asset(asset['id'], user.get('id'))
                                if success:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                        else:
                            st.info("Your listing")
    else:
        st.info("No assets available for sale")

def render_auctions():
    st.markdown('<h2>🎯 Auctions</h2>', unsafe_allow_html=True)
    
    user = st.session_state.get('user', {})
    auction_repo = AuctionRepository()
    auction_service = AuctionService()
    asset_repo = AssetRepository()
    
    tab1, tab2 = st.tabs(["🎯 Active Auctions", "➕ Create Auction"])
    
    with tab1:
        auctions = auction_repo.get_active_auctions()
        
        if auctions:
            for auction in auctions:
                end_time = datetime.fromisoformat(auction['end_time'])
                time_left = end_time - datetime.now()
                hours_left = max(0, time_left.total_seconds() / 3600)
                
                with st.expander(f"🔨 {auction['asset_name']} - Current Bid: {auction['current_bid']:,.0f} VC", expanded=True):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown(f"**Auction Type:** {auction['auction_type'].title()}")
                        st.markdown(f"**Starting Price:** {auction['starting_price']:,.0f} VC")
                        st.markdown(f"**Reserve Price:** {auction['reserve_price']:,.0f} VC")
                        st.markdown(f"**Bid Increment:** {auction['bid_increment']:,.0f} VC")
                    
                    with col2:
                        st.markdown(f"**Category:** {auction['category']}")
                        st.markdown(f"**Condition:** {auction['condition']}")
                        
                        if hours_left > 0:
                            st.markdown(f"**⏰ Time Left:** {hours_left:.1f} hours")
                            progress = min(1.0, (1 - hours_left / 24))
                            st.progress(progress)
                        else:
                            st.warning("Auction ending soon!")
                    
                    with col3:
                        if auction['seller_id'] != user.get('id'):
                            min_bid = auction['current_bid'] + auction['bid_increment']
                            bid_amount = st.number_input(
                                "Your Bid (VC)", 
                                min_value=float(min_bid),
                                value=float(min_bid),
                                step=float(auction['bid_increment']),
                                key=f"bid_{auction['id']}"
                            )
                            
                            if st.button("🎯 Place Bid", key=f"place_bid_{auction['id']}", use_container_width=True):
                                success, msg = auction_service.place_bid(auction['id'], user.get('id'), bid_amount)
                                if success:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                        else:
                            st.info("Your auction")
                            if st.button("🔚 End Auction", key=f"end_{auction['id']}"):
                                success, msg = auction_service.finalize_auction(auction['id'])
                                st.info(msg)
                                st.rerun()
        else:
            st.info("No active auctions at the moment")
    
    with tab2:
        st.markdown("### Create New Auction")
        
        my_assets = asset_repo.get_user_assets(user.get('id'))
        eligible_assets = [a for a in my_assets if a['status'] == 'active']
        
        if eligible_assets:
            with st.form("create_auction_form"):
                asset_options = {f"{a['name']} (Value: {a['current_value']:,.0f} VC)": a['id'] for a in eligible_assets}
                selected_asset = st.selectbox("Select Asset to Auction", list(asset_options.keys()))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    auction_type = st.selectbox("Auction Type", ["english", "dutch", "sealed_bid"])
                    starting_price = st.number_input("Starting Price (VC)", min_value=1.0, value=100.0)
                    reserve_price = st.number_input("Reserve Price (VC)", min_value=0.0, value=50.0)
                
                with col2:
                    duration = st.slider("Duration (hours)", 1, 168, 24)
                    bid_increment = st.number_input("Bid Increment (VC)", min_value=1.0, value=10.0)
                
                submit = st.form_submit_button("🚀 Start Auction", use_container_width=True)
                
                if submit:
                    asset_id = asset_options[selected_asset]
                    success, msg = auction_service.create_auction(
                        asset_id=asset_id,
                        seller_id=user.get('id'),
                        auction_type=AuctionType(auction_type),
                        starting_price=starting_price,
                        reserve_price=reserve_price,
                        duration_hours=duration,
                        bid_increment=bid_increment
                    )
                    if success:
                        st.success(f"Auction created! ID: {msg[:8]}...")
                        st.rerun()
                    else:
                        st.error(msg)
        else:
            st.warning("You don't have any eligible assets for auction. Assets must be in 'active' status.")

def render_transactions():
    st.markdown('<h2>💱 Transaction History</h2>', unsafe_allow_html=True)
    
    user = st.session_state.get('user', {})
    transaction_repo = TransactionRepository()
    user_repo = UserRepository()
    
    tab1, tab2 = st.tabs(["📜 My Transactions", "💸 Transfer Coins"])
    
    with tab1:
        transactions = transaction_repo.get_user_transactions(user.get('id'))
        
        if transactions:
            df = pd.DataFrame(transactions)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                total_spent = df[df['from_user_id'] == user.get('id')]['amount'].sum()
                st.metric("Total Spent", f"{total_spent:,.0f} VC")
            with col2:
                total_earned = df[df['to_user_id'] == user.get('id')]['amount'].sum()
                st.metric("Total Earned", f"{total_earned:,.0f} VC")
            with col3:
                st.metric("Net Flow", f"{total_earned - total_spent:,.0f} VC")
            
            st.markdown("---")
            
            for _, tx in df.iterrows():
                is_outgoing = tx['from_user_id'] == user.get('id')
                icon = "🔴" if is_outgoing else "🟢"
                direction = "Sent" if is_outgoing else "Received"
                
                st.markdown(f"""
                <div class="info-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{icon} {direction}</strong> - {tx['type'].upper()}
                            <br><small>{tx['description']}</small>
                        </div>
                        <div style="text-align: right;">
                            <strong style="font-size: 1.2rem; color: {'#EF4444' if is_outgoing else '#10B981'};">
                                {'-' if is_outgoing else '+'}{tx['amount']:,.0f} VC
                            </strong>
                            <br><small>{tx['timestamp'].strftime('%Y-%m-%d %H:%M')}</small>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No transactions yet")
    
    with tab2:
        st.markdown("### Transfer Virtual Coins")
        
        all_users = user_repo.get_all_users()
        other_users = [u for u in all_users if u['id'] != user.get('id')]
        
        if other_users:
            with st.form("transfer_form"):
                user_options = {f"{u['username']} ({u['department']})": u['id'] for u in other_users}
                recipient = st.selectbox("Recipient", list(user_options.keys()))
                amount = st.number_input("Amount (VC)", min_value=1.0, value=100.0)
                description = st.text_input("Description", placeholder="What's this transfer for?")
                
                submit = st.form_submit_button("💸 Send Coins", use_container_width=True)
                
                if submit:
                    current_user = user_repo.get_user_by_id(user.get('id'))
                    if current_user['wallet_balance'] >= amount:
                        recipient_id = user_options[recipient]
                        user_repo.update_wallet(user.get('id'), -amount)
                        user_repo.update_wallet(recipient_id, amount)
                        
                        tx = Transaction(
                            id=str(uuid.uuid4()),
                            type=TransactionType.TRANSFER,
                            from_user_id=user.get('id'),
                            to_user_id=recipient_id,
                            asset_id=None,
                            amount=amount,
                            timestamp=datetime.now(),
                            description=description or "Coin transfer"
                        )
                        transaction_repo.create_transaction(tx)
                        st.success(f"Successfully transferred {amount:,.0f} VC!")
                        st.rerun()
                    else:
                        st.error("Insufficient balance")
        else:
            st.info("No other users to transfer to")

def render_analytics():
    st.markdown('<h2>📊 Business Intelligence Analytics</h2>', unsafe_allow_html=True)
    
    analytics = AnalyticsService()
    load_balancer = LoadBalancer()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "💰 Financial", "🏆 Performance", "🖥️ System Health"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Asset Distribution by Category")
            df = analytics.get_asset_distribution()
            if not df.empty:
                fig = px.pie(df, values='total_value', names='category', 
                            title='Total Value by Category',
                            color_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available")
        
        with col2:
            st.markdown("### Transaction Trends (30 Days)")
            df = analytics.get_transaction_trends(30)
            if not df.empty:
                fig = px.line(df, x='date', y='total_amount', color='type',
                             title='Transaction Volume Over Time')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No transaction data")
    
    with tab2:
        st.markdown("### Coin Circulation Statistics")
        
        circulation = analytics.get_coin_circulation()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Coins", f"{circulation['total_coins_in_circulation']:,.0f}")
        with col2:
            st.metric("Active Users", circulation['active_users'])
        with col3:
            st.metric("Total Transactions", circulation['total_transactions'])
        with col4:
            st.metric("Transaction Volume", f"{circulation['transaction_volume']:,.0f}")
        
        st.markdown("---")
        
        st.markdown("### Department Spending Analysis")
        df = analytics.get_department_spending()
        if not df.empty:
            fig = px.bar(df, x='department', y='total_spent', 
                        color='transaction_count',
                        title='Spending by Department',
                        color_continuous_scale='Viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No spending data")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Top Traders")
            df = analytics.get_top_traders(10)
            if not df.empty:
                fig = px.bar(df, x='username', y=['earned', 'spent'],
                            title='Top 10 Traders by Volume',
                            barmode='group')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No trader data")
        
        with col2:
            st.markdown("### Auction Performance")
            df = analytics.get_auction_performance()
            if not df.empty:
                fig = px.bar(df, x='auction_type', y=['total_auctions', 'successful_auctions'],
                            title='Auction Success by Type',
                            barmode='group')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No auction data")
        
        st.markdown("### Asset Depreciation Report")
        df = analytics.get_asset_depreciation_report()
        if not df.empty:
            st.dataframe(df, use_container_width=True, height=300)
        else:
            st.info("No depreciation data")
    
    with tab4:
        st.markdown("### System Health Dashboard")
        
        health = load_balancer.get_health_status()
        metrics = load_balancer.get_metrics()
        
        status_color = "🟢" if health['status'] == 'healthy' else "🟡"
        st.markdown(f"### {status_color} System Status: {health['status'].upper()}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Connections", health['active_connections'])
        with col2:
            st.metric("Max Connections", health['max_connections'])
        with col3:
            st.metric("Avg Response Time", f"{health['avg_response_time_ms']} ms")
        with col4:
            st.metric("Success Rate", f"{health['success_rate']}%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Request Metrics")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health['active_connections'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Connection Load"},
                gauge={
                    'axis': {'range': [None, health['max_connections']]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, health['max_connections']*0.5], 'color': "lightgreen"},
                        {'range': [health['max_connections']*0.5, health['max_connections']*0.8], 'color': "yellow"},
                        {'range': [health['max_connections']*0.8, health['max_connections']], 'color': "red"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Cache Status")
            cache_status = "🟢 Redis Connected" if health['cache_available'] else "🟡 Local Cache (Fallback)"
            st.info(cache_status)
            
            st.markdown("**Performance Summary:**")
            st.markdown(f"- Total Requests: {metrics['total_requests']:,}")
            st.markdown(f"- Successful: {metrics['successful_requests']:,}")
            st.markdown(f"- Failed: {metrics['failed_requests']:,}")

def render_admin_panel():
    st.markdown('<h2>⚙️ Admin Panel</h2>', unsafe_allow_html=True)
    
    user = st.session_state.get('user', {})
    
    if user.get('role') not in ['admin', 'manager']:
        st.warning("Access restricted to administrators and managers")
        return
    
    disaster_recovery = DisasterRecovery()
    user_repo = UserRepository()
    asset_repo = AssetRepository()
    
    tab1, tab2, tab3, tab4 = st.tabs(["👥 User Management", "📦 Asset Management", "💾 Backup & Recovery", "⚡ System Config"])
    
    with tab1:
        st.markdown("### All Users")
        users = user_repo.get_all_users()
        
        if users:
            df = pd.DataFrame(users)
            st.dataframe(df, use_container_width=True, height=400)
            
            st.markdown("### Coin Allocation")
            with st.form("allocate_coins"):
                user_options = {f"{u['username']}": u['id'] for u in users}
                selected_user = st.selectbox("Select User", list(user_options.keys()))
                amount = st.number_input("Amount to Add/Subtract", value=1000)
                
                col1, col2 = st.columns(2)
                with col1:
                    add = st.form_submit_button("➕ Add Coins", use_container_width=True)
                with col2:
                    subtract = st.form_submit_button("➖ Subtract Coins", use_container_width=True)
                
                if add:
                    user_repo.update_wallet(user_options[selected_user], amount)
                    st.success(f"Added {amount} VC to {selected_user}")
                    st.rerun()
                elif subtract:
                    user_repo.update_wallet(user_options[selected_user], -amount)
                    st.success(f"Subtracted {amount} VC from {selected_user}")
                    st.rerun()
    
    with tab2:
        st.markdown("### All Assets")
        assets = asset_repo.get_all_assets()
        
        if assets:
            df = pd.DataFrame(assets)
            display_cols = ['name', 'category', 'status', 'current_value', 'location', 'condition']
            st.dataframe(df[display_cols], use_container_width=True, height=400)
            
            st.markdown(f"**Total Assets:** {len(assets)}")
            st.markdown(f"**Total Value:** {sum(a['current_value'] for a in assets):,.2f} VC")
    
    with tab3:
        st.markdown("### Backup & Disaster Recovery")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Create Backup")
            if st.button("📸 Create Backup Now", use_container_width=True):
                with st.spinner("Creating backup..."):
                    backup_path = disaster_recovery.create_backup()
                    st.success(f"Backup created: {backup_path}")
            
            if st.button("📄 Export to JSON", use_container_width=True):
                with st.spinner("Exporting..."):
                    export_path = disaster_recovery.export_to_json()
                    st.success(f"Exported to: {export_path}")
            
            st.markdown("#### Auto Backup")
            auto_backup = st.checkbox("Enable Auto Backup (Every Hour)")
            if auto_backup:
                disaster_recovery.start_auto_backup()
                st.info("Auto backup enabled")
            else:
                disaster_recovery.stop_auto_backup()
        
        with col2:
            st.markdown("#### Available Backups")
            backups = disaster_recovery.get_available_backups()
            
            if backups:
                for backup in backups[:10]:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.text(f"{backup['name']} ({backup['size'] / 1024:.1f} KB)")
                    with col_b:
                        if st.button("Restore", key=f"restore_{backup['name']}", use_container_width=True):
                            if disaster_recovery.restore_backup(backup['path']):
                                st.success("Backup restored!")
                            else:
                                st.error("Restore failed")
            else:
                st.info("No backups available")
    
    with tab4:
        st.markdown("### System Configuration")
        
        st.markdown("#### Current Settings")
        settings_data = {
            "Setting": ["Database Path", "Backup Directory", "Max Connections", "Cache TTL", "Max Workers", "Initial Coins"],
            "Value": [Config.DB_PATH, Config.BACKUP_DIR, Config.MAX_CONNECTIONS, Config.CACHE_TTL, Config.MAX_WORKERS, Config.INITIAL_COINS]
        }
        st.table(pd.DataFrame(settings_data))
        
        st.markdown("#### Database Maintenance")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔧 Optimize Database", use_container_width=True):
                with ConnectionPool().get_connection() as conn:
                    conn.execute("VACUUM")
                    conn.execute("ANALYZE")
                st.success("Database optimized")
        
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                cache = CacheManager()
                cache.invalidate_pattern("*")
                st.success("Cache cleared")

# ============== MAIN APPLICATION ==============
def main():
    """Main application entry point"""
    
    initialize_database()
    setup_page_config()
    
    disaster_recovery = DisasterRecovery()
    disaster_recovery.start_auto_backup(interval=3600)
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if not st.session_state['logged_in']:
        render_login_page()
    else:
        menu = render_sidebar()
        
        if "Dashboard" in menu:
            render_dashboard()
        elif "My Assets" in menu:
            render_my_assets()
        elif "Marketplace" in menu:
            render_marketplace()
        elif "Auctions" in menu:
            render_auctions()
        elif "Transactions" in menu:
            render_transactions()
        elif "Analytics" in menu:
            render_analytics()
        elif "Admin Panel" in menu:
            render_admin_panel()

if __name__ == "__main__":
    main()