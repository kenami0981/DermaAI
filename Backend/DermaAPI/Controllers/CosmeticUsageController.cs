using Derma.Application.CosmeticUsage;
using Derma.Domain;
using Derma.Infrastructure;
using MediatR;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace Derma.API.Controllers
{
    public class CosmeticUsageController : BaseApiController
    {
        private readonly IMediator _mediator;
        public CosmeticUsageController(IMediator mediator)
        {
            _mediator = mediator;
        }


        [HttpGet]
        public async Task<ActionResult<List<CosmeticUsage>>> GetCosmeticsUsage() {
            return await _mediator.Send(new CosmeticUsageList.Query());
        }
        [HttpGet("{id}")]
        public async Task<ActionResult<CosmeticUsage>> GetCosmeticUsage(Guid id)
        {
            return await Mediator.Send(new CosmeticUsageDetails.Query { Id = id });
        }
        [HttpPut("{id}")]
        public async Task<IActionResult> EditCosmeticUsage(Guid id, CosmeticUsage cosmeticUsage) {
            cosmeticUsage.Id = id;
            await Mediator.Send(new CosmeticUsageEdit.Command { CosmeticUsage = cosmeticUsage });
            return Ok();
        }
    }
}
