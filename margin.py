class ForexSimulation:
    def __init__(self, initial_deposit, leverage=100):
        self.balance = initial_deposit  # Initial deposit
        self.equity = initial_deposit  # Equity = balance + profit/loss
        self.leverage = leverage       # Account leverage
        self.positions = []            # List of open positions
        self.margin_call_triggered = False

    def calculate_margin(self, lot_size, price):
        """Calculates the margin required for the trade."""
        contract_size = 100000  # Standard lot size in Forex
        return (lot_size * contract_size) / self.leverage

    def open_position(self, lot_size, price, direction):
        """Opens a new position if enough free margin is available."""
        margin_required = self.calculate_margin(lot_size, price)
        free_margin = self.equity - sum(pos["margin"] for pos in self.positions)

        if free_margin < margin_required:
            print(f"Insufficient free margin to open a position. Free margin: {free_margin:.2f}")
            return False

        position = {
            "lot_size": lot_size,
            "entry_price": price,
            "direction": direction,  # "Buy" or "Sell"
            "margin": margin_required,
            "unrealized_pnl": 0.0
        }
        self.positions.append(position)
        print(f"Position opened: {position}")
        return True

    def update_positions(self, current_price):
        """Updates unrealized profit/loss (PnL) for all open positions."""
        total_unrealized_pnl = 0.0
        for position in self.positions:
            lot_size = position["lot_size"]
            if position["direction"] == "Buy":
                position["unrealized_pnl"] = (current_price - position["entry_price"]) * lot_size * 100000
            elif position["direction"] == "Sell":
                position["unrealized_pnl"] = (position["entry_price"] - current_price) * lot_size * 100000
            total_unrealized_pnl += position["unrealized_pnl"]

        self.equity = self.balance + total_unrealized_pnl
        self.check_margin_call()

    def close_position(self, position_index):
        """Closes an open position and realizes its profit/loss."""
        position = self.positions.pop(position_index)
        realized_pnl = position["unrealized_pnl"]
        self.balance += realized_pnl
        print(f"Position closed. Realized PnL: {realized_pnl:.2f}, New Balance: {self.balance:.2f}")
        self.update_positions(current_price=position["entry_price"])  # Recalculate equity after closing

    def check_margin_call(self):
        """Checks if equity has dropped below margin requirements and alerts."""
        total_margin = sum(pos["margin"] for pos in self.positions)
        if self.equity < total_margin:
            self.margin_call_triggered = True
            print(f"⚠️ Margin Call! Equity: {self.equity:.2f}, Total Margin: {total_margin:.2f}")

    def summary(self):
        """Prints a summary of the account and positions."""
        print(f"Balance: {self.balance:.2f}, Equity: {self.equity:.2f}")
        print(f"Current Positions: {len(self.positions)}")
        for i, pos in enumerate(self.positions, 1):
            print(f"  {i}. {pos}")
        if self.margin_call_triggered:
            print("⚠️ Margin call triggered! Close positions or deposit more funds.")

# Example Usage
if __name__ == "__main__":
    # Initialize simulation with $1000 deposit and 1:100 leverage
    sim = ForexSimulation(initial_deposit=1000, leverage=100)

    # Open a Buy position of 1 lot at price 1.2000
    sim.open_position(lot_size=0.1, price=1.2000, direction="Buy")

    # Simulate price movement
    prices = [1.2010, 1.2020, 1.1980, 1.1950]
    for price in prices:
        print(f"\nUpdating positions for price: {price}")
        sim.update_positions(current_price=price)
        sim.summary()

    # Attempt to open another position
    sim.open_position(lot_size=0.1, price=1.1950, direction="Sell")

    # Close the first position
    sim.close_position(0)

    # Final summary
    sim.summary()
