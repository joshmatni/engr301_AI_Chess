import pygame
import sys
import threading
import chess                # ← add this!
import time  # Add this import

# Evaluation weights for material balance
tile_scores = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   0
}


# UI Constants
 #Updated UI Constants
WIDTH, HEIGHT = 600, 600  # Increase screen size
DIMENSION = 8  # Chessboard is 8x8
SQ_SIZE = HEIGHT // DIMENSION  # Adjust square size based on new screen size
MAX_FPS = 15  # for animations and input handling
IMAGES = {}

import os

def load_images():
    """Load images for pieces into the global IMAGES dict using your Chess_* filenames"""
    # where “ai_chess_engine.py” lives
    base_dir   = os.path.dirname(__file__)
    images_dir = os.path.join(base_dir, 'images')

    # your files are named like Chess_plt60.png, Chess_pdt60.png, etc.
    pieces     = ['p','n','b','r','q','k']
    suffix_map = {'w': 'lt',   # white pieces use 'lt'
                  'b': 'dt'}   # black pieces use 'dt'

    for color in ['w','b']:
        for p in pieces:
            fname = f"Chess_{p}{suffix_map[color]}60.png"
            path  = os.path.join(images_dir, fname)

            # key needs to match how you assemble it elsewhere: e.g. 'wP', 'bK'
            key = color + p.upper()
            IMAGES[key] = pygame.transform.scale(
                pygame.image.load(path),
                (SQ_SIZE, SQ_SIZE)
            )


def draw_board(screen):
    """Draw chessboard squares on the screen."""
    colors = [pygame.Color('white'), pygame.Color('gray')]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[(r + c) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_grid_labels(screen):
    """Draw grid labels (letters and numbers) around the chessboard."""
    font = pygame.font.SysFont('Arial', 20)
    letters = 'ABCDEFGH'
    numbers = '87654321'

    # Draw letters (A-H) along the bottom
    for i in range(DIMENSION):
        letter = font.render(letters[i], True, pygame.Color('black'))
        screen.blit(letter, (i * SQ_SIZE + SQ_SIZE // 2 - letter.get_width() // 2, HEIGHT - 20))

    # Draw numbers (1-8) along the left side
    for i in range(DIMENSION):
        number = font.render(numbers[i], True, pygame.Color('black'))
        screen.blit(number, (5, i * SQ_SIZE + SQ_SIZE // 2 - number.get_height() // 2))

def draw_pieces(screen, board):
    """Draw pieces on their current squares"""
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # convert chess.py square to row, col for pygame
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            key = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().upper()
            screen.blit(IMAGES[key], pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def evaluate(board):
    """Improved evaluation function based on material count, repetition penalty, and positional bonuses"""
    score = 0

    # Material evaluation
    for piece_type, value in tile_scores.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value

    # Positional bonus for controlling the center
    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    for square in center_squares:
        if board.piece_at(square):
            if board.piece_at(square).color == chess.WHITE:
                score += 0.1
            else:
                score -= 0.1

    # Penalize repeated positions
    if board.is_repetition():
        score -= 0.5 if board.turn == chess.WHITE else -0.5

    return score

def minimax(board, depth, alpha, beta, maximizing_player):
    """Minimax with alpha-beta pruning"""
    if depth == 0 or board.is_game_over():
        return evaluate(board)
    if maximizing_player:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def find_best_move(board, depth):
    """Find the best move for the current player using minimax"""
    best_move = None
    best_value = -float('inf') if board.turn == chess.WHITE else float('inf')
    for move in board.legal_moves:
        board.push(move)
        board_value = minimax(board, depth - 1, -float('inf'), float('inf'), board.turn == chess.BLACK)
        board.pop()
        if board.turn == chess.WHITE:
            if board_value > best_value:
                best_value = board_value
                best_move = move
        else:
            if board_value < best_value:
                best_value = board_value
                best_move = move
    return best_move

# Flag to indicate AI is thinking
ai_thinking = False

import time  # Ensure this is imported at the top

def ai_move(board, depth=3):
    """Threaded AI move to avoid blocking the UI"""
    global ai_thinking, ai_last_move
    print("AI is calculating its move...")  # Debugging statement
    move = find_best_move(board, depth)
    if move:
        print(f"AI chose move: {move}")  # Debugging statement
        board.push(move)
        ai_last_move = move  # Store the AI's last move
    else:
        print("AI could not find a valid move.")  # Debugging statement
        ai_last_move = None
    ai_thinking = False

def draw_highlight(screen, square, board):
    """Draw a border around the selected square. Red if capturing an opponent's piece."""
    col = chess.square_file(square)
    row = 7 - chess.square_rank(square)

    # Check if the square contains an opponent's piece
    piece = board.piece_at(square)
    if piece and piece.color != board.turn:  # Opponent's piece
        color = pygame.Color('red')
    else:
        color = pygame.Color('yellow')

    pygame.draw.rect(
        screen,
        color,
        pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE),
        3  # Border thickness
    )

def draw_legal_moves(screen, board, selected_square):
    """Highlight the legal moves for the selected piece."""
    if selected_square is None:
        return

    # Iterate through all legal moves
    for move in board.legal_moves:
        if move.from_square == selected_square:
            to_square = move.to_square
            col = chess.square_file(to_square)
            row = 7 - chess.square_rank(to_square)

            # Check if the move captures an opponent's piece
            if board.piece_at(to_square):
                color = pygame.Color('red')  # Red for capturing moves
            else:
                color = pygame.Color('green')  # Green for normal moves

            pygame.draw.rect(
                screen,
                color,
                pygame.Rect(col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE),
                3  # Border thickness
            )

# def draw_buttons(screen):
#     """Draw Reset button on the screen."""
#     font = pygame.font.SysFont('Arial', 20)

#     # Reset button
#     reset_button = pygame.Rect(10, 10, 100, 30)
#     pygame.draw.rect(screen, pygame.Color('lightgray'), reset_button)
#     reset_text = font.render("Reset", True, pygame.Color('black'))
#     screen.blit(reset_text, (reset_button.x + 25, reset_button.y + 5))

#     return reset_button


def give_hint(board):
    """Provide a hint for the player by suggesting a good move."""
    if board.turn == chess.WHITE:
        hint_move = find_best_move(board, depth=2)  # Use a lower depth for quick hints
        if hint_move:
            print(f"Hint: Try move {hint_move.uci()}")
            return hint_move
    print("No hints available.")
    return None

def display_message(screen, message, color):
    """Display a message in the center of the screen."""
    font = pygame.font.SysFont('Arial', 36)
    text_surface = font.render(message, True, color)
    screen.blit(
        text_surface,
        (WIDTH // 2 - text_surface.get_width() // 2, HEIGHT // 2 - text_surface.get_height() // 2)
    )
    pygame.display.flip()
    time.sleep(2)  # Display the message for 2 seconds

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    board = chess.Board()
    load_images()
    selected_square = None
    player_clicks = []  # Track clicks for move
    global ai_thinking, ai_last_move
    ai_last_move = None  # Initialize the AI's last move

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                location = pygame.mouse.get_pos()

                # Check if Reset button is clicked
                if reset_button.collidepoint(location):
                    board = chess.Board()  # Reset the board
                    selected_square = None
                    player_clicks = []
                    ai_last_move = None
                    ai_thinking = False
                    print("Game reset!")

                # Check if Hint button is clicked
                elif hint_button.collidepoint(location):
                    hint_move = give_hint(board)
                    if hint_move:
                        font = pygame.font.SysFont('Arial', 20)
                        hint_text = f"Hint: {hint_move.uci()}"
                        text_surface = font.render(hint_text, True, pygame.Color('blue'))
                        screen.blit(text_surface, (10, 50))  # Display hint below the buttons
                        pygame.display.flip()
                        time.sleep(2)  # Display the hint for 2 seconds

                # Handle chessboard clicks
                elif not ai_thinking:
                    col = location[0] // SQ_SIZE
                    row = location[1] // SQ_SIZE
                    clicked_square = chess.square(col, 7 - row)
                    if selected_square is None:
                        # First click
                        if board.piece_at(clicked_square) and board.piece_at(clicked_square).color == chess.WHITE:
                            selected_square = clicked_square
                            player_clicks = [selected_square]
                    else:
                        # Second click
                        player_clicks.append(clicked_square)
                        move = chess.Move(player_clicks[0], player_clicks[1])
                        if move in board.legal_moves:
                            board.push(move)
                            print(f"Player made move: {move}")  # Debugging statement
                            ai_thinking = True
                            threading.Thread(target=ai_move, args=(board, 3)).start()
                        selected_square = None
                        player_clicks = []

        # Check for game over conditions
        if board.is_checkmate():
            display_message(screen, "You Win!", pygame.Color('green'))
            board = chess.Board()  # Reset the board after the message
        elif board.is_stalemate():
            display_message(screen, "Stalemate!", pygame.Color('orange'))
            board = chess.Board()  # Reset the board after the message

        # Draw the board and pieces
        draw_board(screen)
        draw_pieces(screen, board)

        # Draw grid labels
        draw_grid_labels(screen)

        # Draw Reset and Hint buttons
        reset_button, hint_button = draw_buttons(screen)

        # Highlight the selected square
        if selected_square is not None:
            draw_highlight(screen, selected_square, board)
            # Highlight legal moves for the selected piece
            draw_legal_moves(screen, board, selected_square)

        # Display AI thinking message or AI's last move
        if ai_thinking:
            font = pygame.font.SysFont('Arial', 24)
            text_surface = font.render("AI is thinking...", True, pygame.Color('red'))
            screen.blit(text_surface, (WIDTH // 2 - text_surface.get_width() // 2, HEIGHT // 2 - text_surface.get_height() // 2))
        elif ai_last_move:
            # Temporarily display the AI's last move in the middle
            font = pygame.font.SysFont('Arial', 24)
            move_text = f"AI moved: {ai_last_move.uci()}"
            text_surface = font.render(move_text, True, pygame.Color('blue'))
            screen.blit(text_surface, (WIDTH // 2 - text_surface.get_width() // 2, HEIGHT // 2 - text_surface.get_height() // 2))
            pygame.display.flip()
            time.sleep(2)  # Delay for 2 seconds
            ai_last_move = None  # Clear the move after displaying it

        pygame.display.flip()
        clock.tick(MAX_FPS)

def draw_buttons(screen):
    """Draw Reset and Hint buttons on the screen."""
    font = pygame.font.SysFont('Arial', 20)

    # Reset button
    reset_button = pygame.Rect(10, 10, 100, 30)
    pygame.draw.rect(screen, pygame.Color('lightgray'), reset_button)
    reset_text = font.render("Reset", True, pygame.Color('black'))
    screen.blit(reset_text, (reset_button.x + 25, reset_button.y + 5))

    # Hint button
    hint_button = pygame.Rect(120, 10, 100, 30)
    pygame.draw.rect(screen, pygame.Color('lightgray'), hint_button)
    hint_text = font.render("Hint", True, pygame.Color('black'))
    screen.blit(hint_text, (hint_button.x + 30, hint_button.y + 5))

    return reset_button, hint_button

if __name__ == '__main__':
    main()