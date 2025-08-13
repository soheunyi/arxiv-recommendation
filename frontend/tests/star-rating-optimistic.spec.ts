import { test, expect } from '@playwright/test';

test.describe('Star Rating Optimistic Updates', () => {
  test.beforeEach(async ({ page }) => {
    // Mock the API endpoints
    await page.route('**/api/papers/search**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          papers: [
            {
              id: 'test-paper-1',
              title: 'Test Paper for Rating',
              authors: ['John Doe', 'Jane Smith'],
              abstract: 'This is a test paper abstract for testing star rating functionality.',
              category: 'cs.AI',
              published_date: '2024-01-15',
              arxiv_url: 'https://arxiv.org/abs/2401.00001',
              pdf_url: 'https://arxiv.org/pdf/2401.00001.pdf',
              current_score: 0.85,
              rating: 0
            }
          ],
          total: 1,
          hasMore: false
        })
      });
    });

    await page.route('**/api/ratings/user**', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([])
      });
    });

    // Navigate to search page
    await page.goto('/search');
    
    // Perform a search to load papers
    await page.fill('input[placeholder*="Search"]', 'test paper');
    await page.click('button[type="submit"]');
    
    // Wait for search results to load
    await page.waitForSelector('[data-testid="paper-card"]', { timeout: 5000 });
  });

  test('should provide immediate visual feedback when clicking stars', async ({ page }) => {
    // Mock successful rating update
    await page.route('**/api/ratings/update**', async (route) => {
      // Add delay to simulate network request
      await new Promise(resolve => setTimeout(resolve, 1000));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'rating-1',
          paper_id: 'test-paper-1',
          rating: 4,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        })
      });
    });

    const paperCard = page.locator('[data-testid="paper-card"]').first();
    const starRating = paperCard.locator('[data-testid="star-rating"]');
    
    // Initially, no stars should be filled
    const filledStars = starRating.locator('[data-testid="star-filled"]');
    await expect(filledStars).toHaveCount(0);
    
    // Click on the 4th star
    const fourthStar = starRating.locator('button').nth(3);
    await fourthStar.click();
    
    // Immediately after click, 4 stars should be filled (optimistic update)
    await expect(starRating.locator('[data-testid="star-filled"]')).toHaveCount(4);
    
    // Stars should show updating state (pulse animation)
    await expect(starRating.locator('button').first()).toHaveClass(/animate-pulse/);
    
    // Wait for API call to complete and verify stars remain filled
    await page.waitForTimeout(1200);
    await expect(starRating.locator('[data-testid="star-filled"]')).toHaveCount(4);
    
    // Updating state should be cleared
    await expect(starRating.locator('button').first()).not.toHaveClass(/animate-pulse/);
  });

  test('should rollback optimistic update on API failure', async ({ page }) => {
    // Mock failed rating update
    await page.route('**/api/ratings/update**', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 500));
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal server error' })
      });
    });

    const paperCard = page.locator('[data-testid="paper-card"]').first();
    const starRating = paperCard.locator('[data-testid="star-rating"]');
    
    // Click on the 3rd star
    const thirdStar = starRating.locator('button').nth(2);
    await thirdStar.click();
    
    // Immediately after click, 3 stars should be filled (optimistic update)
    await expect(starRating.locator('[data-testid="star-filled"]')).toHaveCount(3);
    
    // Wait for API call to fail
    await page.waitForTimeout(800);
    
    // Stars should rollback to original state (0 filled)
    await expect(starRating.locator('[data-testid="star-filled"]')).toHaveCount(0);
    
    // Error toast should be visible
    await expect(page.locator('.toast').filter({ hasText: 'Failed to update rating' })).toBeVisible();
  });

  test('should handle clearable rating functionality', async ({ page }) => {
    // First, set a rating
    await page.route('**/api/ratings/update**', async (route) => {
      const request = route.request();
      const postData = await request.postDataJSON();
      
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'rating-1',
          paper_id: 'test-paper-1',
          rating: postData.rating,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        })
      });
    });

    const paperCard = page.locator('[data-testid="paper-card"]').first();
    const starRating = paperCard.locator('[data-testid="star-rating"]');
    
    // Click on the 3rd star to set rating
    await starRating.locator('button').nth(2).click();
    await page.waitForTimeout(200);
    
    // Should have 3 filled stars
    await expect(starRating.locator('[data-testid="star-filled"]')).toHaveCount(3);
    
    // Click on the 3rd star again to clear rating
    await starRating.locator('button').nth(2).click();
    
    // Should immediately show 0 filled stars (optimistic clear)
    await expect(starRating.locator('[data-testid="star-filled"]')).toHaveCount(0);
  });

  test('should disable interaction during update', async ({ page }) => {
    // Mock slow rating update
    await page.route('**/api/ratings/update**', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 2000));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'rating-1',
          paper_id: 'test-paper-1',
          rating: 2,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        })
      });
    });

    const paperCard = page.locator('[data-testid="paper-card"]').first();
    const starRating = paperCard.locator('[data-testid="star-rating"]');
    
    // Click on the 2nd star
    await starRating.locator('button').nth(1).click();
    
    // During update, stars should be disabled
    const starButtons = starRating.locator('button');
    await expect(starButtons.first()).toBeDisabled();
    
    // Try clicking another star during update - should not change rating
    await starRating.locator('button').nth(3).click({ force: true });
    
    // Should still show 2 filled stars, not 4
    await expect(starRating.locator('[data-testid="star-filled"]')).toHaveCount(2);
    
    // Wait for update to complete
    await page.waitForTimeout(2200);
    
    // Stars should be enabled again
    await expect(starButtons.first()).toBeEnabled();
  });

  test('should show hover effects when not updating', async ({ page }) => {
    const paperCard = page.locator('[data-testid="paper-card"]').first();
    const starRating = paperCard.locator('[data-testid="star-rating"]');
    
    // Hover over the 3rd star
    await starRating.locator('button').nth(2).hover();
    
    // Should show hover state for first 3 stars
    await expect(starRating.locator('[data-testid="star-filled"]')).toHaveCount(3);
    
    // Hover feedback text should appear
    await expect(page.locator('text=Rate 3 stars')).toBeVisible();
    
    // Move mouse away
    await starRating.locator('button').nth(2).hover({ force: false });
    await page.mouse.move(0, 0);
    
    // Hover state should clear
    await expect(starRating.locator('[data-testid="star-filled"]')).toHaveCount(0);
  });

  test('should not show hover effects when updating', async ({ page }) => {
    // Mock slow rating update
    await page.route('**/api/ratings/update**', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 1500));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'rating-1',
          paper_id: 'test-paper-1',
          rating: 1,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        })
      });
    });

    const paperCard = page.locator('[data-testid="paper-card"]').first();
    const starRating = paperCard.locator('[data-testid="star-rating"]');
    
    // Click on the 1st star to start update
    await starRating.locator('button').nth(0).click();
    
    // Try to hover during update
    await starRating.locator('button').nth(3).hover();
    
    // Should still show only 1 filled star (from click), not hover effect
    await expect(starRating.locator('[data-testid="star-filled"]')).toHaveCount(1);
    
    // No hover feedback should appear during update
    await expect(page.locator('text=Rate 4 stars')).not.toBeVisible();
  });

  test('should maintain keyboard accessibility', async ({ page }) => {
    const paperCard = page.locator('[data-testid="paper-card"]').first();
    const starRating = paperCard.locator('[data-testid="star-rating"]');
    const thirdStar = starRating.locator('button').nth(2);
    
    // Focus the third star with tab navigation
    await thirdStar.focus();
    
    // Should have focus outline
    await expect(thirdStar).toBeFocused();
    
    // Press Enter to rate
    await page.keyboard.press('Enter');
    
    // Should immediately show optimistic update
    await expect(starRating.locator('[data-testid="star-filled"]')).toHaveCount(3);
    
    // Press Space should also work for rating
    const fifthStar = starRating.locator('button').nth(4);
    await fifthStar.focus();
    await page.keyboard.press('Space');
    
    // Should show 5 filled stars
    await expect(starRating.locator('[data-testid="star-filled"]')).toHaveCount(5);
  });
});