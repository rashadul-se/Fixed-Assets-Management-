# 🏢 Fixed Assets Management System (FAMS)

> **A comprehensive enterprise-grade Fixed Assets Management System with Virtual Coin Economy, featuring real-time auctions, marketplace trading, analytics dashboard, and disaster recovery capabilities.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [User Guide](#user-guide)
- [Admin Guide](#admin-guide)
- [API Documentation](#api-documentation)
- [Security Features](#security-features)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The **Fixed Assets Management System (FAMS)** is a modern, enterprise-ready solution designed to revolutionize how organizations track, manage, and trade their fixed assets. Built with Python and Streamlit, FAMS introduces an innovative **Virtual Coin (VC) Economy** that enables internal asset trading, auctions, and inter-departmental transfers.

### Why FAMS?

Traditional asset management systems are often rigid, costly, and lack flexibility. FAMS addresses these challenges by:

- **🎮 Gamifying Asset Management**: Virtual coins create engagement and accountability
- **💹 Enabling Internal Trading**: Marketplace and auction systems for asset redistribution
- **📊 Providing Real-time Analytics**: Business intelligence dashboards for informed decisions
- **🔒 Ensuring Enterprise Security**: Military-grade encryption and authentication
- **🚀 Scaling Effortlessly**: Connection pooling, caching, and load balancing built-in
- **💾 Guaranteeing Data Safety**: Automated backups and disaster recovery

---

## ✨ Key Features

### 🏪 Virtual Coin Economy
- **Initial Allocation**: Each user receives 10,000 Virtual Coins (VC) upon registration
- **Peer-to-Peer Transfers**: Send coins to colleagues with transaction history
- **Market Dynamics**: Buy, sell, and trade assets using virtual currency
- **Economic Analytics**: Track coin circulation, spending patterns, and wealth distribution

### 📦 Asset Management
- **Comprehensive Tracking**: Name, category, location, condition, depreciation rate
- **QR Code Generation**: Unique identifiers for physical asset tracking
- **Status Management**: Active, maintenance, disposed, for sale, auction
- **Multi-Category Support**: Electronics, furniture, vehicles, equipment, real estate, software
- **Depreciation Calculation**: Automatic value adjustment based on configured rates
- **Search & Filter**: Advanced search with category and status filters

### 🎯 Auction System
- **Multiple Auction Types**:
  - **English Auction**: Classic ascending bid format
  - **Dutch Auction**: Descending price mechanism
  - **Sealed Bid**: Anonymous competitive bidding
- **Real-time Bidding**: Live updates and countdown timers
- **Reserve Price Protection**: Sellers set minimum acceptable prices
- **Bid Increment Control**: Customizable minimum bid increases
- **Automatic Settlement**: Winner determination and coin transfer
- **Auction Analytics**: Success rates and performance metrics

### 🛒 Marketplace
- **Asset Listings**: Simple buy-now functionality for quick sales
- **Advanced Search**: Keyword, category, and price filtering
- **Dynamic Sorting**: Price (low-high, high-low) and date sorting
- **Instant Purchase**: One-click buying with balance verification
- **Transaction History**: Complete audit trail for all purchases

### 💱 Transaction Management
- **Complete History**: View all incoming and outgoing transactions
- **Transaction Types**: Purchase, sale, auction win, transfer, allocation
- **Financial Metrics**: Total spent, earned, and net flow calculations
- **Coin Transfers**: P2P virtual coin transfers with descriptions
- **Real-time Updates**: Instant balance and transaction updates

### 📊 Business Intelligence & Analytics
- **Asset Distribution**: Visual breakdowns by category and department
- **Transaction Trends**: 30-day rolling analysis with charts
- **Department Spending**: Comparative spending analytics across departments
- **Top Traders**: Leaderboard of most active users
- **Auction Performance**: Success rates by auction type
- **Depreciation Reports**: Comprehensive asset value tracking
- **Coin Circulation**: Economic health monitoring
- **Interactive Visualizations**: Plotly charts and graphs

### ⚙️ Admin Panel
- **User Management**: 
  - View all registered users
  - Add/subtract virtual coins
  - Monitor user activity and balances
- **Asset Oversight**: 
  - Complete asset inventory
  - Total value calculations
  - Status monitoring
- **Backup & Recovery**:
  - One-click manual backups
  - Automated hourly backups
  - JSON export functionality
  - Point-in-time recovery
  - Backup history with sizes
- **System Configuration**:
  - Database optimization
  - Cache management
  - Connection pool monitoring
  - Performance tuning

### 🖥️ System Health Monitoring
- **Connection Metrics**: Active connections and capacity
- **Response Time Tracking**: Average response times in milliseconds
- **Success Rate Monitoring**: Request success/failure ratios
- **Cache Status**: Redis connectivity and fallback status
- **Load Balancer Health**: Real-time system health indicators
- **Visual Gauges**: Interactive connection load visualization

---

## 🏗️ System Architecture

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                       │
│                   (Streamlit Web Interface)                  │
├─────────────────────────────────────────────────────────────┤
│   Dashboard  │  Assets  │  Marketplace  │  Auctions  │ Admin│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      BUSINESS LOGIC LAYER                    │
├─────────────────────────────────────────────────────────────┤
│  AuthService  │  AssetService  │  AuctionService  │ Analytics│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA ACCESS LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  UserRepo  │  AssetRepo  │  AuctionRepo  │  TransactionRepo │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ INFRASTRUCTURE│    │  PERSISTENCE │    │   CACHING    │
├──────────────┤    ├──────────────┤    ├──────────────┤
│ Load Balancer│    │   SQLite DB  │    │ Redis Cache  │
│ Conn Pool    │    │   WAL Mode   │    │ (+ Fallback) │
│ Security Mgr │    │   Indexed    │    │  TTL: 300s   │
│ Disaster Rec │    │   ACID       │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Component Breakdown

#### 1. **Presentation Layer** (Streamlit)
- User interface components
- Form handling and validation
- Real-time updates and rerun management
- Session state management
- Responsive design with columns and tabs

#### 2. **Business Logic Layer**
- **AuthService**: User registration, login, authentication
- **AssetService**: Asset creation, listing, purchasing
- **AuctionService**: Auction creation, bidding, finalization
- **AnalyticsService**: Data aggregation and reporting

#### 3. **Data Access Layer** (Repository Pattern)
- **BaseRepository**: Abstract base with common operations
- **UserRepository**: User CRUD operations
- **AssetRepository**: Asset management and queries
- **AuctionRepository**: Auction lifecycle management
- **TransactionRepository**: Financial transaction logging

#### 4. **Infrastructure Layer**
- **SecurityManager**: Password hashing, encryption, sanitization
- **ConnectionPool**: Thread-safe database connection management
- **CacheManager**: Redis/local cache with automatic fallback
- **LoadBalancer**: Request distribution and health monitoring
- **DisasterRecovery**: Backup automation and restoration

#### 5. **Persistence Layer**
- **SQLite Database**: ACID-compliant relational storage
- **WAL Mode**: Write-Ahead Logging for concurrency
- **Indexed Tables**: Optimized query performance
- **Normalized Schema**: 4 core tables with foreign keys

#### 6. **Caching Layer**
- **Redis**: Primary distributed cache (optional)
- **Local Cache**: In-memory fallback with TTL
- **Smart Invalidation**: Pattern-based cache clearing
- **Pickle Serialization**: Complex object storage

---

## 🛠️ Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Framework** | Streamlit | 1.28.0 | Web interface and UI |
| **Language** | Python | 3.8+ | Backend logic |
| **Database** | SQLite | 3.x | Data persistence |
| **Cache** | Redis | 5.0.1 | Performance optimization |
| **Data Analysis** | Pandas | 2.1.1 | Data manipulation |
| **Visualization** | Plotly | 5.17.0 | Interactive charts |

### Key Libraries

- **Security**: hashlib (PBKDF2-HMAC), uuid
- **Concurrency**: threading, queue, concurrent.futures
- **Serialization**: pickle, json
- **Compression**: gzip
- **Date/Time**: datetime, python-dateutil

### Design Patterns

- **Repository Pattern**: Data access abstraction
- **Singleton Pattern**: Connection pool, cache manager
- **Service Layer Pattern**: Business logic separation
- **Factory Pattern**: Object creation management
- **Observer Pattern**: Real-time UI updates

---

## 📥 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Redis (optional, for enhanced performance)
- Git (for cloning repository)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/fixed-assets-management.git
cd fixed-assets-management
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Redis (Optional)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**macOS (Homebrew):**
```bash
brew install redis
brew services start redis
```

**Windows:**
```bash
# Download from: https://github.com/microsoftarchive/redis/releases
# Or use Docker:
docker run -d -p 6379:6379 redis:latest
```

**Docker (Any OS):**
```bash
docker run -d -p 6379:6379 --name redis-fams redis:alpine
```

### Step 5: Verify Installation

```bash
python -c "import streamlit; import redis; import pandas; print('All dependencies installed!')"
```

---

## 🚀 Quick Start

### 1. Start the Application

```bash
streamlit run complete_fixed_assets.py
```

The application will open in your browser at `http://localhost:8501`

### 2. Create Admin Account

1. Navigate to the **Register** tab
2. Fill in the registration form:
   ```
   Username: admin
   Password: Admin@123
   Confirm Password: Admin@123
   Role: admin
   Department: IT
   ```
3. Click **Register**
4. You'll receive 10,000 VC automatically

### 3. Login

1. Switch to **Login** tab
2. Enter your credentials
3. Click **Login**
4. You'll be redirected to the dashboard

### 4. Create Your First Asset

1. Go to **My Assets** → **Add New Asset** tab
2. Enter asset details:
   ```
   Asset Name: Dell Laptop XPS 15
   Category: Electronics
   Initial Cost: 1500
   Location: Office Floor 3
   Description: High-performance laptop
   Condition: New
   Depreciation Rate: 20%
   Department: IT
   ```
3. Click **Create Asset**

### 5. List Asset for Sale

1. Go to **My Assets** → **Asset List** tab
2. Find your asset and expand it
3. Set sale price (e.g., 1400 VC)
4. Click **List for Sale**

### 6. Create Test Users

Register additional users to test marketplace and auctions:
```
User 1: john_doe / User@123 / user / Finance
User 2: jane_smith / User@456 / user / Operations
```

### 7. Test Marketplace

1. Login as a different user
2. Go to **Marketplace**
3. Find the listed asset
4. Click **Buy Now**

### 8. Create an Auction

1. Login as asset owner
2. Go to **My Assets**
3. Select an active asset
4. Go to **Auctions** → **Create Auction**
5. Configure auction parameters
6. Click **Start Auction**

---

## 📖 User Guide

### Dashboard Overview

The dashboard provides a comprehensive overview of your assets and activity:

- **My Assets**: Total count of assets you own
- **Total Asset Value**: Combined value of all your assets in VC
- **Transactions**: Number of transactions you've participated in
- **Wallet Balance**: Current virtual coin balance

### Managing Assets

#### Creating Assets
1. Navigate to **My Assets** → **Add New Asset**
2. Fill required fields (marked with *)
3. Set depreciation rate (0-50%)
4. Submit form

#### Listing for Sale
1. Go to **My Assets** → **Asset List**
2. Expand asset card
3. Enter sale price
4. Click **List for Sale**

#### Asset Status Lifecycle
```
Active → For Sale → Sold (transferred to buyer)
Active → Auction → Auctioned (transferred to winner)
Active → Maintenance → Active
Active → Disposed (end of life)
```

### Using the Marketplace

#### Searching Assets
- Use keyword search for names/descriptions
- Filter by category
- Sort by price or date

#### Purchasing Assets
1. Browse marketplace listings
2. Click on asset to view details
3. Verify price and condition
4. Click **Buy Now** (must have sufficient balance)
5. Asset transfers immediately to your inventory

### Participating in Auctions

#### Auction Types

**English Auction (Ascending)**
- Starts at starting price
- Bidders compete by increasing bid
- Highest bid at end wins
- Must meet reserve price

**Dutch Auction (Descending)**
- Starts high, decreases over time
- First bidder wins at current price
- Fast-paced, strategic timing

**Sealed Bid**
- Submit maximum bid privately
- All bids revealed at end
- Highest bidder wins

#### Placing Bids
1. Go to **Auctions** → **Active Auctions**
2. Review auction details
3. Enter bid amount (min = current + increment)
4. Click **Place Bid**
5. Receive confirmation

#### Winning Auctions
- System automatically transfers asset
- Coins deducted from winner
- Coins credited to seller
- Transaction recorded

### Managing Transactions

#### Viewing History
- Go to **Transactions** → **My Transactions**
- See all incoming/outgoing transactions
- View financial metrics (spent/earned/net)

#### Transferring Coins
1. Go to **Transactions** → **Transfer Coins**
2. Select recipient
3. Enter amount
4. Add description
5. Click **Send Coins**

---

## 🔧 Admin Guide

### User Management

#### Viewing All Users
- Go to **Admin Panel** → **User Management**
- View table with all user data
- Monitor balances and activity

#### Allocating/Deducting Coins
1. Select user from dropdown
2. Enter amount
3. Click **Add Coins** or **Subtract Coins**
4. Use for:
   - Bonus rewards
   - Penalty deductions
   - Initial allocations
   - Balance corrections

### Asset Oversight

#### Monitoring Assets
- View all assets across organization
- Track total inventory value
- Monitor asset distribution
- Identify underutilized assets

### Backup & Recovery

#### Manual Backups
- Click **Create Backup Now**
- Compressed .gz file created
- Stored in `/backups` directory
- Includes timestamp in filename

#### Automated Backups
- Enable **Auto Backup** checkbox
- Runs every hour (configurable)
- Keeps last 50 backups
- Automatic cleanup of old backups

#### Restoring Backups
1. View available backups list
2. Note backup size and date
3. Click **Restore** button
4. System restores database
5. Application restarts required

#### JSON Export
- Click **Export to JSON**
- Creates human-readable export
- Useful for:
  - Data migration
  - External analysis
  - Audit reports
  - Integration with other systems

### System Maintenance

#### Database Optimization
- Click **Optimize Database**
- Runs VACUUM (reclaims space)
- Runs ANALYZE (updates statistics)
- Improves query performance

#### Cache Management
- Click **Clear Cache**
- Invalidates all cached data
- Forces fresh database queries
- Use after bulk updates

### System Configuration

View current settings:
- Database path
- Backup directory
- Max connections (100)
- Cache TTL (300 seconds)
- Max workers (50)
- Initial coins (10,000)

---

## 🔒 Security Features

### Authentication & Authorization

- **Password Hashing**: PBKDF2-HMAC-SHA256 with 100,000 iterations
- **Unique Salts**: Per-user salt for rainbow table protection
- **Role-Based Access**: admin, manager, user, auditor roles
- **Session Management**: Streamlit session state for user tracking

### Input Validation

- **SQL Injection Prevention**: Parameterized queries throughout
- **Input Sanitization**: Removal of dangerous characters
- **Type Validation**: Strict type checking on all inputs
- **Length Limits**: Maximum field lengths enforced

### Data Protection

- **Encryption**: XOR-based encryption for sensitive data
- **No Plain Storage**: Passwords never stored in plain text
- **Audit Trail**: Complete transaction history
- **Data Integrity**: Foreign key constraints and ACID compliance

### Network Security

- **Local by Default**: Runs on localhost
- **Redis Authentication**: Support for password-protected Redis
- **Connection Limits**: Max 100 concurrent connections
- **Rate Limiting**: Built-in request throttling

---

## ⚡ Performance Optimization

### Connection Pooling

- Pre-allocated pool of 10 connections
- Grows to 100 maximum
- Timeout-based acquisition
- Automatic connection recycling

### Caching Strategy

- **Redis Primary**: Distributed caching for multiple instances
- **Local Fallback**: In-memory cache when Redis unavailable
- **Smart TTL**: 300-second default, 60-second for volatile data
- **Pattern Invalidation**: Efficient cache clearing

### Database Optimization

- **WAL Mode**: Write-Ahead Logging for concurrent access
- **Strategic Indexes**: On status, owner_id, timestamp fields
- **Query Optimization**: Limit clauses and specific column selection
- **Prepared Statements**: Parameterized query reuse

### Load Balancing

- ThreadPoolExecutor with 50 workers
- Health monitoring and metrics
- Automatic request distribution
- Graceful degradation

---

## 🐛 Troubleshooting

### Common Issues

#### Application Won't Start

**Error**: `ModuleNotFoundError`
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Error**: `Address already in use`
```bash
# Solution: Kill existing process or use different port
streamlit run complete_fixed_assets.py --server.port 8502
```

#### Login Issues

**Problem**: "Invalid credentials"
- Verify username spelling (case-sensitive)
- Check password matches registration
- Ensure user is active in database

**Problem**: Can't register
- Username already exists
- Password too short (<6 characters)
- Passwords don't match

#### Redis Connection

**Warning**: "Redis unavailable"
- System automatically uses local cache
- Install Redis for better performance
- Check Redis is running: `redis-cli ping`

#### Performance Issues

**Slow queries**
```bash
# Run database optimization
# From Admin Panel → System Config → Optimize Database
```

**High memory usage**
```bash
# Clear cache
# From Admin Panel → System Config → Clear Cache
```

### Database Corruption

If database becomes corrupted:

```bash
# 1. Stop application
# 2. Restore from backup
python -c "
from complete_fixed_assets import DisasterRecovery
dr = DisasterRecovery()
backups = dr.get_available_backups()
dr.restore_backup(backups[0]['path'])
"
# 3. Restart application
```

### Logs and Debugging

Enable debug mode:
```bash
streamlit run complete_fixed_assets.py --logger.level=debug
```

Check Streamlit logs:
```bash
# Logs typically in ~/.streamlit/
cat ~/.streamlit/logs/*.log
```

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open Pull Request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints
- Write unit tests for new features
- Update documentation

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=complete_fixed_assets

# Run specific test
pytest tests/test_auth.py::test_login
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Fixed Assets Management System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🗺️ Roadmap

### Version 2.0 (Q1 2025)
- [ ] Mobile responsive design
- [ ] REST API for external integrations
- [ ] Email notifications
- [ ] Advanced reporting (PDF/Excel export)
- [ ] Multi-language support

### Version 2.1 (Q2 2025)
- [ ] Two-factor authentication (2FA)
- [ ] Asset images upload
- [ ] Barcode scanner integration
- [ ] Advanced depreciation models
- [ ] Department budgets

### Version 3.0 (Q3 2025)
- [ ] Multi-tenant support
- [ ] Cloud deployment options
- [ ] Real-time chat
- [ ] AI-powered asset valuation
- [ ] Predictive maintenance

---

## 🙏 Acknowledgments

- **Streamlit Team** - For the amazing web framework
- **Redis Labs** - For high-performance caching
- **Plotly** - For beautiful visualizations
- **SQLite Team** - For reliable database engine
- **Open Source Community** - For inspiration and support

---
