using Derma.Application.CosmeticUsage;
using Derma.Application.SkinAnalysis;
using Derma.Domain;
using Derma.Infrastructure;
using MediatR;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Security.Claims;

namespace Derma.API.Controllers
{
    
    public class CosmeticUsageController : BaseApiController
    {
        private readonly IMediator _mediator;
        private Guid CurrentUserId =>
   Guid.TryParse(User.FindFirstValue(ClaimTypes.NameIdentifier), out var id) ? id : Guid.Empty;
        private bool IsAdmin => User.IsInRole("Admin");
        public CosmeticUsageController(IMediator mediator)
        {
            _mediator = mediator;
        }


        [HttpGet]
        [Authorize(AuthenticationSchemes = "Bearer,ApiKey")]
        public async Task<ActionResult<List<CosmeticUsage>>> GetCosmeticsUsage() {
            return await _mediator.Send(new CosmeticUsageList.Query { UserId = CurrentUserId, IsAdmin = IsAdmin });
        }
        [HttpGet("{id}")]
        [Authorize(AuthenticationSchemes = "Bearer,ApiKey")]
        public async Task<ActionResult<CosmeticUsage>> GetCosmeticUsage(Guid id)
        {
            return await Mediator.Send(new CosmeticUsageDetails.Query { Id = id, UserId = CurrentUserId, IsAdmin = IsAdmin });
        }
        [HttpPost]
        [Authorize]
        public async Task<IActionResult> CreateCosmeticUsage(CosmeticUsage cosmeticUsage)
        {
            await Mediator.Send(
                new CosmeticUsageCreate.Command { CosmeticUsage = cosmeticUsage });

            return Ok();
        }
        [HttpPut("{id}")]
        [Authorize]
        public async Task<IActionResult> EditCosmeticUsage(Guid id, CosmeticUsage cosmeticUsage) {
            cosmeticUsage.Id = id;
            await Mediator.Send(new CosmeticUsageEdit.Command { CosmeticUsage = cosmeticUsage, UserId = CurrentUserId, IsAdmin = IsAdmin });
            return Ok();
        }
        [HttpDelete("{id}")]
        [Authorize]
        public async Task<IActionResult> DeleteCosmeticUsage(Guid id)
        {
            await Mediator.Send(
                new CosmeticUsageDelete.Command {Id = id, UserId = CurrentUserId, IsAdmin = IsAdmin });

            return Ok();
        }
    }
}
